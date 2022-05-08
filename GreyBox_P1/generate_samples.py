from generator_pretrained import *
import os
import torch
from movie_test import *
from torchvision.io import write_video as wv

seed = 272
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


device = 'cuda'


gen = generator()
gan = gen.genfunc()

if not os.path.exists(f'./saved_videos'):
	os.mkdir(f'./saved_videos')


## Samples to generate
num_samples = 10800
with torch.no_grad():
	for i in range(num_samples):
	## Saving video batches 
	outputs = gen.batchno(1,gan).to(device)
	video = outputs[0]
	wv(f'./saved_videos/file_{i}.mp4', 255*video.to('cpu'),32)




