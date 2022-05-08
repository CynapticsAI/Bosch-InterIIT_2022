# (32,3,64,64)
# (10,32,3,64,64)
import sys

sys.path.append("./VST")
import torch
import torchvision
import os
from VST import get_inference as f


def get_probabs(input_tensor):
    # input tensor has shape (batch_sz,32,3,64,64)
    outputs = []
    for i in range(input_tensor.size()[0]):
        aa = input_tensor[i].permute([0, 3, 2, 1])
        x = aa
        if not (os.path.exists("./temp/")):
            os.mkdir("./temp/")
        torchvision.io.write_video(f"temp/tmp{1}.mp4", x, 32)
        probs = f.get_logits(f"temp/tmp{1}.mp4")
        outputs.append([mm[1] for mm in probs])
    res = torch.FloatTensor(outputs)
    class_label = torch.argmax(res, dim=1)
    return class_label
