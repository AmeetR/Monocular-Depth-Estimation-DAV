"""
We implement the decoder of the depth attention volume network. This is the decoder implemented from https://arxiv.org/pdf/2004.02760.pdf.

It's a simple decoder with bilinear upsampling, convolutional layers, and batch normalization. The output of the decoder is the depth map.

The decoder part of our network contains a straightforward up-scaling scheme
that increases the spatial dimension from 29×38 to 57×76 and then to 114×152.
Upsampling consists of two bilinear interpolation layers followed by convolutional
layers with a kernel size of 3 × 3. Two convolutional layers with a kernel size of
5 × 5 are then used to estimate the final depth map.

Input Operations k s d CH RES Output
layer8-Y bilinear+conv+bn 3 1 1 256 57 × 76 up-conv-1
up-conv-1 bilinear+conv+bn 3 1 1 128 114 × 152 up-conv-2
up-conv-2 conv+bn+relu 5 1 1 64 114 × 152 refine-1
refine-1 conv 5 1 1 1 114 × 152 depth

"""


import torch.nn as nn
import math



class Decoder(nn.Module):
    def __init__(self): 
        super(Decoder, self).__init__()
        self.bilinear_upsample_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128)
        )
        self.bilinear_upsample_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        x = self.bilinear_upsample_1(x)
        x = self.bilinear_upsample_2(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x


