from PIL import Image
import cv2
from DDIM import DDIM_Sampler_Reverse,DDIM_sampler
from preProcess import getX_t
import torch
import numpy as np
from torchvision import transforms
IMAGE_PATH='./Robustness sample/sample_image.png'
modelConfig = {
            "batch_size": 1,
            "T": 1000,
            "channel": 128,
            "channel_mult": [1, 2, 3, 4],
            "attn": [2],
            "num_res_blocks": 2,
            "multiplier": 2.,
            "beta_1": 1e-4,
            "beta_T": 0.02,
            "img_size": 64,
            "grad_clip": 1.,
            "device": "cuda:0",
            "save_weight_dir": "./Checkpoints/",
            "test_load_weight": "ckpt_165_.pt",
            "DDIM_step":200,
        }
transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
def compute_acc(secret,recover):
    acc=0
    for i in range(3):
        for j in range(64):
            for k in range(64):
                if secret[0][i][j][k]<= 0 and recover[0][i][j][k]<=0:
                    acc+=1
                elif secret[0][i][j][k]> 0 and recover[0][i][j][k]>0:
                    acc+=1

    return acc/(3*64*64)

def image_generation():
    x_t=getX_t().reshape((1, 3, 64, 64))
    np.save('./Robustness sample/secret.npy',x_t)
    x_t=torch.tensor(x_t,device=modelConfig['device'])
    x_0=DDIM_sampler(modelConfig,x_t)
    return x_0


def jpeg_compression(noise,Q):
    # 打开图像文件
    image = Image.open(IMAGE_PATH)
    # 将图像压缩为JPEG格式
    image.save('./Robustness sample/compressed_image.jpg', 'JPEG', quality=Q)
    c_img=cv2.imread('./Robustness sample/compressed_image.jpg')
    img=np.zeros([1,3,64,64])
    img[0][0]=c_img[:,:,0]
    img[0][1]=c_img[:,:,1]
    img[0][2]=c_img[:,:,2]
    img=img/255
    img=(img-0.5)/0.5
    img=torch.tensor(img,device=modelConfig['device'],dtype=torch.float32)
    reverse=DDIM_Sampler_Reverse(modelConfig,img)
    reverse=reverse.cpu().data.numpy()
    acc=compute_acc(noise,reverse)
    print(acc)

def Gaussian_noise(noise,sigma):
    #读取图片
    img = cv2.imread(IMAGE_PATH)
    img = img/255
    #设置高斯分布的均值和方差
    mean = 0
    Sigma=sigma
    #根据均值和标准差生成符合高斯分布的噪声
    gauss = np.random.normal(mean,Sigma,(64,64,3))
    #给图片添加高斯噪声
    noisy_img = img + gauss
    #设置图片添加高斯噪声之后的像素值的范围
    noisy_img = np.clip(noisy_img,a_min=0,a_max=1)
    #保存图片
    cv2.imwrite('./Robustness sample/g_noise.png',noisy_img*255)

    c_img=cv2.imread('./Robustness sample/g_noise.png')
    img=np.zeros([1,3,64,64])
    img[0][0]=c_img[:,:,0]
    img[0][1]=c_img[:,:,1]
    img[0][2]=c_img[:,:,2]
    img=img/255
    img=(img-0.5)/0.5
    img=torch.tensor(img,device=modelConfig['device'],dtype=torch.float32)
    reverse=DDIM_Sampler_Reverse(modelConfig,img)
    reverse=reverse.cpu().data.numpy()
    acc=compute_acc(noise,reverse)
    print(acc)

#x_0=image_generation()
secret=np.load('./Robustness sample/secret.npy')
#jpeg_compression(secret,80)
Gaussian_noise(secret,0.05)
