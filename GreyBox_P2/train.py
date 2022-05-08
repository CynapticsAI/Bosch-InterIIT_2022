import torch.optim as optim
import torch
from resnet import *
import torch.nn.functional as FN
import random
import os
import glob
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import StepLR

seed = 272
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

path = str(os.getcwd())


def train(student, device,epoch, samples = 200, batch_size = 16):


    path_to_tensors = glob.glob(path + '/saved_tensors/*')[:samples]  # dd path of saved tensors
    path_to_preds = glob.glob(path+'/movieN_preds/*')[:samples] 
    

    # loading probabs   
    victim_preds_list =[]
    for x in path_to_preds:
        victim_preds_list.append(torch.load(x).type(torch.LongTensor).to(device))


    zipper=list(zip(path_to_tensors,victim_preds_list))

    random.shuffle(zipper)
    path_to_tensors,victim_preds_list = zip(*zipper)

    
    student.train()
    optimizer_S = optim.Adam(student.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler_S = StepLR(optimizer_S, step_size=30, gamma=0.1)
    for i, x in enumerate(tqdm(path_to_tensors)):
        input_batch = torch.load(x)  # shape = (16,16,128,128,3) -> (batch_sz, F, W, H, C)
        victim_preds = victim_preds_list[i]
        optimizer_S.zero_grad()
        s_logit = student(input_batch.permute((0, 4, 1, 2, 3)))
        criterion = FN.cross_entropy
        loss_S = criterion(s_logit, victim_preds, reduction="mean")
        loss_S.backward()
        optimizer_S.step()
        scheduler_S.step(loss_S)

        print(f"Iteration = {i} \t Loss = {loss_S}")

    # checkpointing model   
    torch.save(student, path + f'/checkpoint/GreyBox_P2_{epoch}')
    return student


if not os.path.exists(path + '/checkpoint'):
    os.mkdir(path + '/checkpoint')



device = 'cuda'
student = get_vid_resnet(num_classes=600).to(device=device)
num_epochs = 100
samples_taken = 200

for i in range(num_epochs):
    print(f"----- Epoch {i} -----")
    student=train(student, device,i,samples_taken)
