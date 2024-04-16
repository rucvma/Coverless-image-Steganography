import os
from typing import Dict
import torch
import torch.optim as optim
import glob
from PIL import Image
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from Diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from Model import UNet
from torch.optim.lr_scheduler import _LRScheduler

class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, warm_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = warm_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = None
        self.base_lrs = None
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]


    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)
        
class ImageDataSet(Dataset):
    def __init__(self, data,transforms=None):
        self.data = data
        self.transforms = transforms

    def __getitem__(self, index):
        img_path=self.data[index]
        img = cv2.imread(img_path)
        img=Image.fromarray(img)
        if self.transforms is not None:
            img = self.transforms(img)
        return img
    def __len__(self):
        return len(self.data)

def get_data():
    dataset_path='../data/*.png'
    imageFiles = glob.glob(dataset_path)
    
    return imageFiles
def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    # dataset
    '''
    dataset = CIFAR10(
        root='./CIFAR10', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    '''
    #[0.70070046 0.6006601  0.5894853 ] [0.258258   0.27078328 0.2421064 ]
    data=get_data()
    transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset=ImageDataSet(data,transform)
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    # model setup
    trained_epoch=-1
    net_model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["training_load_weight"]), map_location=device))
        trained_epoch=15
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"])
    # 设置学习率衰减，按余弦函数的1/2个周期衰减，从``lr``衰减至0
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    # 设置逐步预热调度器，学习率从0逐渐增加至multiplier * lr，共用1/10总epoch数，后续学习率按``cosineScheduler``设置进行变化
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)
    # 实例化训练模型
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    # start training
    for e in range(trained_epoch+1,modelConfig["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images in tqdmDataLoader:
                # train
                optimizer.zero_grad()                                    # 清空过往梯度
                x_0 = images.to(device)                                  # 将输入图像加载到计算设备上
                loss = trainer(x_0).sum() / 1000.                        # 前向传播并计算损失
                loss.backward()                                          # 反向计算梯度
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])    # 裁剪梯度，防止梯度爆炸
                optimizer.step()                                         # 更新参数
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })                                                       # 设置进度条显示内容
        warmUpScheduler.step()                                           # 调度器更新学习率
        if e%5==0:
            torch.save(net_model.state_dict(), os.path.join(
                modelConfig["save_weight_dir"], 'ckpt_' + str(e) + "_.pt"))  # 保存模型


def eval(modelConfig: Dict):
    # load model and evaluate
    with torch.no_grad():
        # 建立和加载模型
        device = torch.device(modelConfig["device"])
        model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)
        ckpt = torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        # 实例化反向扩散采样器
        model.eval()
        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
        # Sampled from standard normal distribution
        # 随机生成高斯噪声图像并保存
        noisyImage = torch.randn(
            size=[modelConfig["batch_size"], 3, 64, 64], device=device)
        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        save_image(saveNoisy, os.path.join(
            modelConfig["sampled_dir"], modelConfig["sampledNoisyImgName"]), nrow=modelConfig["nrow"])
        # 反向扩散并保存输出图像
        sampledImgs = sampler(noisyImage)
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        sampledImgs=sampledImgs.cpu()
        sampledImgs=sampledImgs.data.numpy()
        sampledImgs_convert=np.zeros_like(sampledImgs)
        sampledImgs_convert[:,0,:,:]=sampledImgs[:,2,:,:]
        sampledImgs_convert[:,1,:,:]=sampledImgs[:,1,:,:]
        sampledImgs_convert[:,2,:,:]=sampledImgs[:,0,:,:]
        sampledImgs_convert=torch.from_numpy(sampledImgs_convert)
        save_image(sampledImgs_convert, os.path.join(
            modelConfig["sampled_dir"],  modelConfig["sampledImgName"]), nrow=modelConfig["nrow"])