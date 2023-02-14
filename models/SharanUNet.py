import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.geometry import conv_soft_argmax2d
from kornia.filters import gaussian_blur2d
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
import numpy as np


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, drop=0.2):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            # Here batchnorm comes after activation, but another variant is
            # it can come before activation - change this later to have an option
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.ELU(inplace=True),
            nn.BatchNorm2d(mid_channels),

            nn.Dropout(p=drop),

            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.ELU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, drop=0.2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, drop=drop)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, drop=0.2):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, drop=drop)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, drop=drop)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return torch.sigmoid(self.conv(x))


class Model(nn.Module):
    def __init__(self, config={'in_channels': 1, 'out_channels': 2, 'features': [64, 128, 256, 512], 'batch_size': 4}, device="cuda", kernel=torch.ones(3, 3), bilinear=True):
        super(Model, self).__init__()
        self.n_channels = config['in_channels']
        self.n_classes = config['out_channels']
        self.bilinear = bilinear
        self.kernel = kernel.to(device)

        self.inc = DoubleConv(self.n_channels, 16, drop=0.3)
        self.down1 = Down(16, 32, drop=0.37)
        self.down2 = Down(32, 64, drop=0.43)
        self.down3 = Down(64, 128, drop=0.5)
        factor = 2 if bilinear else 1
        self.down4 = Down(128, 256 // factor, drop=0.4)
        self.up1 = Up(256, 128 // factor, bilinear, drop=0.5)
        self.up2 = Up(128, 64 // factor, bilinear, drop=0.43)
        self.up3 = Up(64, 32 // factor, bilinear, drop=0.37)
        self.up4 = Up(32, 16, bilinear, drop=0.3)
        self.outc = OutConv(16, self.n_classes)
        self.gauss = gaussian_blur2d
        self.sftargmax = conv_soft_argmax2d

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        logits = self.gauss(logits, (3, 3), sigma=(1, 1))
        binary_coords, binary = self.sftargmax(logits, (3, 3), (1, 1), (1, 1), output_value=True)
        return binary, logits