import numpy as np
import matplotlib.pyplot as plt
import math

def getX_t(input_size):
    label=np.random.randint(0,2,size=input_size)
    zeros_index=[]
    ones_index=[]
    for i in range(0,3*64*64):
        if label[i]==0:
            zeros_index.append(i)
        else:
            ones_index.append(i)
    x_t=label.astype(np.float32)
    #print(len(zeros_index))
    #print(len(ones_index))
    for _ in range(0,int(0.81*len(zeros_index))):
        temp=np.random.choice(zeros_index)
        zeros_index.pop(zeros_index.index(temp))
        x_t[temp]=np.random.random()-1.1
    for _ in range(0,int(0.73*len(zeros_index))):
        temp=np.random.choice(zeros_index)
        zeros_index.pop(zeros_index.index(temp))
        x_t[temp]=np.random.random()-2.0
    for _ in range(0,len(zeros_index)):
        temp=np.random.choice(zeros_index)
        zeros_index.pop(zeros_index.index(temp))
        x_t[temp]=np.random.random()-3.0
    for _ in range(0,int(0.81*len(ones_index))):
        temp=np.random.choice(ones_index)
        ones_index.pop(ones_index.index(temp))
        x_t[temp]=np.random.random()+0.1
    for _ in range(0,int(0.73*len(ones_index))):
        temp=np.random.choice(ones_index)
        ones_index.pop(ones_index.index(temp))
        x_t[temp]=np.random.random()+1.0
    for _ in range(0,len(ones_index)):
        temp=np.random.choice(ones_index)
        ones_index.pop(ones_index.index(temp))
        x_t[temp]=np.random.random()+2.0
    #print(np.mean(x_t))
    #print(np.var(x_t))
    return x_t

def show_his(x_t):
    plt.hist(x_t,bins=[-3,-2.5,-2,-1.5,-1,0,1,1.5,2,2.5,3,3.5])
    plt.xlabel('Value')
    plt.ylabel('Rate')
    plt.show()

def text_to_bin(datapath):
    Bytes=[]
    bin_string=np.zeros((3*64*64),dtype=np.int32)
    start=0
    end=8
    with open(datapath,'rb+') as file:
        data=file.read()
    for i in data:
        Bytes.append(i)
    Bytes.append(3)
    if len(Bytes)>(3*64*64/8):
        raise("Text length too big")
    #b=bytes(Bytes)
    #a=bytes.decode(b)
    for item in Bytes:
        temp=list(bin(item)[2:])
        for _ in range(8-len(temp)):
            temp.insert(0,0)
        bin_string[start:end]=np.array(temp)
        start=start+8
        end=end+8
    return bin_string

def bin_to_text(bin_string):
    recover_bytes=[]

    for i in range(0,3*64*64,8):
        temp=''.join(map(str,bin_string[i:i+8]))  
        temp=int(temp,2)
        if temp!=3:
            recover_bytes.append(temp)
        else:
            break
    recover_bytes=bytes(recover_bytes)
    return bytes.decode(recover_bytes)

#getX_t()