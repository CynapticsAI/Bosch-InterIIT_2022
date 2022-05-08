import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as FN
import scipy.linalg
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.models as models
from time import time
from torch.autograd import Variable
from Generator import Generator
#from vid_resnet import *
from get_swint_preds import *
from resnet import *


def estimate_gradient_objective(surrogate_model, x, epsilon=1e-7, m=1, device='cuda'):
    with torch.no_grad():
        # Sample unit noise vector
        N = x.size(0)  # batch_size
        F = x.size(1)  # number of frames
        C = x.size(2)  # channels
        S = x.size(3)  # image dimensions (height or width)
        dim = S ** 2 * C * F

        u = np.random.randn(N * m * dim).reshape(-1, m, dim)  # generate random points from normal distribution
        d = np.sqrt(np.sum(u ** 2, axis=2)).reshape(-1, m, 1)  # map to a uniform distribution on a unit sphere
        u = torch.Tensor(u / d).view(-1, m, C, F, S, S)
        u = torch.cat((u, torch.zeros(N, 1, C, F, S, S)), dim=1)  # Shape N, m + 1, S^2
        u = u.view(-1, m + 1, C, F, S, S)

        evaluation_points = (x.view(-1, 1, C, F, S, S).cpu() + epsilon * u).view(-1, C, F, S, S)
        evaluation_points = torch.tanh(evaluation_points)

        # Compute the approximation sequentially to allow large values of m
        pred_victim = []
        pred_surrogate = []
        max_number_points = 32 * 156  # Hardcoded value to split the large evaluation_points tensor to fit in GPU

        #print(evaluation_points.shape)

        for i in (range(N * m // max_number_points + 1)):
            pts = evaluation_points[i * max_number_points: (i + 1) * max_number_points]
            pts = pts.to(device)  # TODO: check this
            pred_victim_pts = get_probabs(pts, device=device).detach()  # wrapper
            # print(pts.permute((0, 2, 1, 3, 4)).shape)
            pred_surrogate_pts = surrogate_model(pts.permute((0, 2, 1, 3, 4)))
            pred_victim.append(pred_victim_pts)
            pred_surrogate.append(pred_surrogate_pts)

        pred_victim = torch.cat(pred_victim, dim=0).to(device)
        pred_surrogate = torch.cat(pred_surrogate, dim=0).to(device)
        u = u.to(device)

        criterion = FN.cross_entropy
        loss_values = -criterion(pred_surrogate, pred_victim, reduction="none").view(-1, m + 1)
        differences = loss_values[:, :-1] - loss_values[:, -1].view(-1, 1)  # (batch_size, 1)
        differences = differences.view(-1, m, 1, 1, 1, 1)

        # Formula for Forward Finite Differences
        gradient_estimates = 1 / epsilon * differences * u[:, :-1]
        gradient_estimates *= dim  # the shape of gradient_estimates must be equal to input
        gradient_estimates = gradient_estimates.mean(dim=1).view(-1, F, C, S, S)

        surrogate_model.train()
        loss_G = loss_values[:, -1].mean()
        return gradient_estimates.detach(), loss_G


def compute_gradient(surrogate_model, x, device="cpu"):
    surrogate_model.eval()
    N = x.size(0)
    x_copy = x.clone().detach().requires_grad_(True)
    x_ = x_copy.to(device).permute(0, 2, 1, 3, 4)

    pred_victim = get_probabs(x_).detach()  # wrapper
    pred_surrogate = surrogate_model(x_.permute((0, 2, 1, 3, 4)))
    criterion = FN.cross_entropy
    loss_values = - criterion(pred_surrogate, torch.log(pred_victim))#, reduction="mean")
    loss_values.backward()
    surrogate_model.train()
    return x_copy.grad, loss_values


# x = Variable(torch.randn([5, 3, 1, 64, 64]).cpu())
# model = Generator()
# x_out = model(x)
#
# victim_target_label = torch.tensor([10])
# surrogate_logit = torch.rand(1, 400)
# model = get_vid_resnet(num_classes=400)
# estimate_gradient_objective(model, x_out, m=1)
# compute_gradient(model, x_out)
