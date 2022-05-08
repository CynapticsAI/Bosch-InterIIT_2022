import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class GeneratorA(nn.Module):
    # nz = batch_size,
    def __init__(self, nz=256, ngf=64, nc=3, img_size=64, activation=None, final_bn=True):
        super(GeneratorA, self).__init__()

        if activation is None:
            raise ValueError("Provide a valid activation function")
        self.activation = activation

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * 2 * self.init_size ** 2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(ngf * 2),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        if final_bn:
            self.conv_blocks2 = nn.Sequential(
                nn.Conv2d(ngf * 2, ngf, 3, stride=1, padding=1),
                nn.BatchNorm2d(ngf),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
                # nn.Tanh(),
                nn.BatchNorm2d(nc, affine=False)
            )
        else:
            self.conv_blocks2 = nn.Sequential(
                nn.Conv2d(ngf * 2, ngf, 3, stride=1, padding=1),
                nn.BatchNorm2d(ngf),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
                # nn.Tanh(),
                # nn.BatchNorm2d(nc, affine=False)
            )

    def forward(self, z, pre_x=False):
        out = self.l1(z.view(z.shape[0], -1))
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks2(img)
        # img = torch.tanh(img)
        # img = (img + 1) * 0.5
        img = torch.clamp(img, -0.5, 0.5) + 0.5

        # print(f"Gradient shape {img.shape}")
        if pre_x:
            return img.unsqueeze(1).repeat(1, 32, 1, 1, 1).permute(0, 2, 1, 3, 4)
        else:
            # img = nn.functional.interpolate(img, torch.Size([256, 32, 3, 32, 32])=2)
            return img.unsqueeze(1).repeat(1, 32, 1, 1, 1).permute(0, 2, 1, 3, 4)


class GeneratorC(nn.Module):
    '''
    Conditional Generator
    '''

    def __init__(self, nz=100, num_classes=10, ngf=64, nc=1, img_size=32):
        super(GeneratorC, self).__init__()

        self.label_emb = nn.Embedding(num_classes, nz)

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz * 2, ngf * 2 * self.init_size ** 2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(ngf * 2),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(nc, affine=False)
        )

    def forward(self, z, label):
        # Concatenate label embedding and image to produce input
        label_inp = self.label_emb(label)
        gen_input = torch.cat((label_inp, z), -1)

        out = self.l1(gen_input.view(gen_input.shape[0], -1))
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks2(img)
        return img


# batch_size = 16
# nz = 256
#
# z = torch.randn((batch_size, nz)).to('cpu')
# z.requires_grad = True
# gat = GeneratorA(nz=256, activation=torch.tanh)
# op = gat(z)
# print(op.size())
#
# gat1 = GeneratorC(nz=nz, num_classes=400)
# op1 = gat(z)
# torch.mean(op1).backward()
# print(op1.size())
# print(z.grad.shape)
#
