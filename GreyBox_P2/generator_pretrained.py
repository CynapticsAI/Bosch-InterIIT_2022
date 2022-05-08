import os
import torch
import math
import numpy as np
from torchvision.io import read_video, read_video_timestamps
from videogpt import download, load_vqvae, load_videogpt
from videogpt.data import preprocess
class generator():
    def __init__(self):
        self.device = torch.device('cuda')
        
    def genfunc(self):
        ROOT = 'pretrained_models'
        gpt = load_videogpt('ucf101_uncond_gpt', device=self.device).to(self.device)
        return gpt

    def batchno(self,numb,model):
        samples = model.sample(numb)
        b, c, t, h, w = samples.shape
        samples = samples.permute(0, 2, 3, 4, 1)
        #samples = (samples.cpu().numpy() * 255).astype('uint8')
        return samples

#genfunc = generator()
##gan=genfunc.genfunc()
#gan_output=genfunc.batchno(1,gan)
#print(gan_output)
#np.save("ab.npy",gan_output)
