from torchvision.models.video.resnet import r2plus1d_18
import torch


def get_vid_resnet(num_classes):  # Takes(batchsz,3,32,h,h)
    model = r2plus1d_18(num_classes=num_classes)
    return model
