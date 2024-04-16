import torch
import os
from typing import Dict
from Model import UNet
import numpy as np
import cv2
import torch.nn.functional as F
from tqdm import tqdm
from Diffusion import GaussianDiffusionSampler
from preProcess import getX_t
from torchvision.utils import save_image
def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    # ``torch.gather``的用法建议看https://zhuanlan.zhihu.com/p/352877584的第一条评论
    # 在此处的所有调用实例中，v都是一维，可以看作是索引取值，即等价v[t], t大小为[batch_size, 1]
    out = torch.gather(v, index=t, dim=0).float().to(device)
    # 再把索引到的值reshape到[batch_size, 1, 1, ...], 维度和x_shape相同
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

def DDIM_sampler(modelConfig: Dict,noise=None,num=0):
    # load model and evaluate
    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        betas=torch.linspace(modelConfig['beta_1'], modelConfig['beta_T'], modelConfig["T"],device=device).double()
        alphas = 1. - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        #alphas_bar_prev = torch.tensor(np.append(1.0, alphas_bar[:-1].cpu().data.numpy()),device=device)
        #alphas_bar_next = np.append(alphas_bar[1:].cpu().data.numpy(), 0.0)
        model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)
        ckpt = torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)#不能删！！
        print("model load weight done.")
        # 实例化反向扩散采样器
        model.eval()

        ddim_timesteps=modelConfig["DDIM_step"]
        #ddim_timesteps=num
        c=int(modelConfig["T"]/ddim_timesteps)
        ddim_timestep_seq = np.asarray(list(range(0, modelConfig["T"], c)))
        ddim_timestep_seq=ddim_timestep_seq+1

        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])
        if noise==None:
            x_t = torch.randn(size=[modelConfig["batch_size"], 3, 64, 64], device=device)
        else:
            x_t = noise
        for i in tqdm(reversed(range(0, ddim_timesteps)), desc='sampling loop time step', total=ddim_timesteps):
            t = x_t.new_ones([x_t.shape[0], ], dtype=torch.long) * ddim_timestep_seq[i]
            prev_t = x_t.new_ones([x_t.shape[0], ], dtype=torch.long) * ddim_timestep_prev_seq[i]
            #1
            alphas_bar_t = extract(alphas_bar, t, x_t.shape)
            alphas_bar_t_pre = extract(alphas_bar, prev_t, x_t.shape)
            #2
            noise_predict = model(x_t,t)
            #3 predicted x_0
            pred_x0 = (x_t - torch.sqrt((1. - alphas_bar_t)) * noise_predict) / torch.sqrt(alphas_bar_t)
            pred_x0 = torch.clip(pred_x0, -1, 1)  
            #4 direction pointing to x_t
            pred_dir_xt = torch.sqrt(1 - alphas_bar_t_pre) * noise_predict

            #5 compute the complete formula of Eq(12) in paper
            x_prev=torch.sqrt(alphas_bar_t_pre) * pred_x0 + pred_dir_xt
            x_t = x_prev
        x_0 = x_t * 0.5 + 0.5  # [0 ~ 1]
        x_0=x_0.cpu()
        x_0=x_0.data.numpy()
        merged = cv2.merge([x_0[0][0], x_0[0][1], x_0[0][2]])
        #cv2.imwrite('../Stego_valid/'+str(num)+'.png',merged*255)
        #cv2.imwrite('../sample/'+str(num)+'.png',merged*255)
        return merged

def DDIM_Sampler_Reverse(modelConfig: Dict,im):
    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        betas=torch.linspace(modelConfig['beta_1'], modelConfig['beta_T'], modelConfig["T"],device=device).double()
        alphas = 1. - betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                        num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)
        ckpt = torch.load(os.path.join(
                modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        sampler = GaussianDiffusionSampler(
                model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

        print("model load weight done.")
        # 实例化反向扩散采样器
        model.eval()

        ddim_timesteps=modelConfig["DDIM_step"]
        c=int(modelConfig["T"]/ddim_timesteps)
        ddim_timestep_seq = np.asarray(list(range(0, modelConfig["T"], c)))
        ddim_timestep_seq=ddim_timestep_seq+1

        ddim_timestep_next_seq = np.append(ddim_timestep_seq[1:],np.array([999]))

        x_t = im

        for i in tqdm(range(0, ddim_timesteps), desc='Reverse loop time step', total=ddim_timesteps):
            t = x_t.new_ones([x_t.shape[0], ], dtype=torch.long) * ddim_timestep_seq[i]
            next_t = x_t.new_ones([x_t.shape[0], ], dtype=torch.long) * ddim_timestep_next_seq[i]
            #1
            alphas_bar_t = extract(alphas_bar, t, x_t.shape)
            alphas_bar_t_next = extract(alphas_bar, next_t, x_t.shape)

            #2
            noise_predict0 = model(x_t,t)
            #3 
            pred_x0 = (x_t - torch.sqrt((1. - alphas_bar_t)) * noise_predict0) / torch.sqrt(alphas_bar_t)
            pred_x0 = torch.clip(pred_x0, -1, 1)  
            #4
            pred_dir_xt = torch.sqrt(1 - alphas_bar_t_next) * noise_predict0

            #5 
            x_next=torch.sqrt(alphas_bar_t_next) * pred_x0 + pred_dir_xt 
            x_t = x_next

        return x_next
def main():

    modelConfig = {
            "epoch": 200,
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
            "nrow": 8,
            "DDIM_step":200,
            "Eta":0
        }
    #x_t=torch.tensor(x_t,device=modelConfig['device'])
    #torch.cuda.manual_seed(2023)
    #x_t = torch.randn(size=[modelConfig["batch_size"], 3, 64, 64], device=modelConfig["device"])
    #x_t = torch.clip(x_t,-2.5,2.5)
    N=1
    for num in range(0,N):
        #sampledImgs=np.zeros([32,3,64,64])
        x_t=getX_t(1*3*64*64).reshape((1, 3, 64, 64))
        x_t=torch.tensor(x_t,device=modelConfig['device'])
        x_0=DDIM_sampler(modelConfig,x_t,0)
        #x_0=DDIM_sampler(modelConfig,x_t,num)
        #save_image(x_0, './tttt.png', nrow=8)
        x_t_reverse=DDIM_Sampler_Reverse(modelConfig,x_0)
        x_t_reverse=x_t_reverse.cpu().data.numpy()
        acc=0
        for i in range(3):
            for j in range(64):
                for k in range(64):
                    if x_t[0][i][j][k]<= 0 and x_t_reverse[0][i][j][k]<=0:
                        acc+=1
                    elif x_t[0][i][j][k]> 0 and x_t_reverse[0][i][j][k]>0:
                        acc+=1
                    #else:
                    #    print('Expect:'+str(x_t[0][i][j][k].cpu().data.numpy())+' Get:'+str(x_t_reverse[0][i][j][k]))
        acc=acc/(3*64*64)
        print(acc)

if __name__ == '__main__':
    main()