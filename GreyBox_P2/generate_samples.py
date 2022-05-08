from generator_pretrained import *
import os
import torch
from movie_test import *

seed = 272
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


device = 'cuda'


gen = generator()
gan = gen.genfunc()

## Enter batch size
batch_sz = 16

if not os.path.exists(f'./saved_tensors'):
	os.mkdir(f'./saved_tensors')

if not os.path.exists(f'./MovieN_preds'):
	os.mkdir(f'./MovieN_preds')


## Samples to generate
num_samples = 200
with torch.no_grad():
	for i in range(num_samples):
	## Saving video batches
	outputs = gen.batchno(batch_sz,gan).to(device)
	torch.save(outputs, f'./saved_tensors/batches_{i}.pt')
	## Saving video outputs
	victim_label = get_probabs(outputs.permute(0,1,4,3,2).detach().cpu(), device=device)
	torch.save(victim_label,f'./MovieN_preds/batches_{i}.pt')
	print(f'Sample number {i} generated')



