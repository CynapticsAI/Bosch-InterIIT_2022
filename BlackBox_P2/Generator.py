import torch.nn as nn
import torch
from torch.autograd import Variable
import os


# Defining layers

def conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)


def conv3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
    return nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)


def deconv2d_first(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(4, 4))


def deconv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=True)


def deconv3d_first(in_channels, out_channels):
    return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(2, 4, 4))


def deconv3d_video(in_channels, out_channels):
    return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(2, 1, 1))


def deconv3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=True)


def batchNorm4d(num_features, eps=1e-5):
    return nn.BatchNorm2d(num_features, eps=eps)


def batchNorm5d(num_features, eps=1e-5):
    return nn.BatchNorm3d(num_features, eps=eps)


def relu(inplace=True):
    return nn.ReLU(inplace)


def lrelu(negative_slope=0.2, inplace=True):
    return nn.LeakyReLU(negative_slope, inplace)


class G_background(nn.Module):
    def __init__(self):
        super(G_background, self).__init__()
        self.model = nn.Sequential(
            deconv2d(1024, 512),
            batchNorm4d(512),
            relu(),
            deconv2d(512, 256),
            batchNorm4d(256),
            relu(),
            deconv2d(256, 128),
            batchNorm4d(128),
            relu(),
            deconv2d(128, 3),
            nn.Tanh()
        )

    def forward(self, x):
        # print('G_background Input =', x.size())
        out = self.model(x)
        # print('G_background Output =', out.size())
        return out


class G_video(nn.Module):
    def __init__(self):
        super(G_video, self).__init__()
        self.model = nn.Sequential(
            deconv3d_video(1024, 1024),
            batchNorm5d(1024),
            relu(),
            deconv3d(1024, 512),
            batchNorm5d(512),
            relu(),
            deconv3d(512, 256),
            batchNorm5d(256),
            relu(),
            deconv3d(256, 128),
            batchNorm5d(128),
            relu(),
        )

    def forward(self, x):
        # print('G_video input =', x.size())
        out = self.model(x)
        # print('G_video output =', out.size())
        return out


class G_encode(nn.Module):
    def __init__(self):
        super(G_encode, self).__init__()
        self.model = nn.Sequential(
            conv2d(3, 128),
            relu(),
            conv2d(128, 256),
            batchNorm4d(256),
            relu(),
            conv2d(256, 512),
            batchNorm4d(512),
            relu(),
            conv2d(512, 1024),
            batchNorm4d(1024),
            relu(),
        )

    def forward(self, x):
        # print('G_encode Input =', x.size())
        out = self.model(x)
        # print('G_encode Output =', out.size())
        return out


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encode = G_encode()
        self.background = G_background()
        self.video = G_video()
        self.gen_net = nn.Sequential(deconv3d(128, 3), nn.Tanh())
        self.mask_net = nn.Sequential(deconv3d(128, 1), nn.Sigmoid())

    def forward(self, x):
        # print('Generator input = ',x.size())
        x = x.squeeze(2)
        # print(x.size())
        encoded = self.encode(x)
        encoded = encoded.unsqueeze(2)
        video = self.video(encoded)
        # print('Video size = ', video.size())

        foreground = self.gen_net(video)
        # print('Foreground size =', foreground.size())

        mask = self.mask_net(video)
        # print('Mask size = ', mask.size())
        mask_repeated = mask.repeat(1, 3, 1, 1, 1)
        # print('Mask repeated size = ', mask_repeated.size())

        x = encoded.view((-1, 1024, x.size()[-1] // 16, x.size()[-1] // 16))
        background = self.background(x)
        # print('Background size = ', background.size())
        background_frames = background.unsqueeze(2).repeat(1, 1, 32, 1, 1)
        out = torch.mul(mask, foreground) + torch.mul(1 - mask, background_frames)
        # print('Generator out = ', out.size())
        v_max = torch.max(out)
        v_min = torch.min(out)
        # out = (out - v_min) / (v_max - v_min)
        out = torch.clamp(out, -0.5, 0.5) + 0.5
        return out

# (batch_sz, channels,1,h,h)
# h = 64
# batch_sz = 10
# x = Variable(torch.randn([batch_sz,3,1,h,h]).cpu())
# model = Generator()
# print(torch.max(model(x)))
# print(torch.min(model(x)))
# print(torch.mean(model(x)))
