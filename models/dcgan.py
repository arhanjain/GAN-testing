from numpy import pad
from torch import nn

import torch
import torchvision

class Discriminator(nn.Module):
    def __init__(self, img_channels, convolution_factor):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # Input: N x Channels x 64 x 64
            nn.Conv2d(in_channels=img_channels, out_channels=1*convolution_factor, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # 32 x 32
            self._block(1*convolution_factor, 2*convolution_factor, 4, 2, 1), # 16 x 16
            self._block(2*convolution_factor, 4*convolution_factor, 4, 2, 1), # 8 x 8
            self._block(4*convolution_factor, 8*convolution_factor, 4, 2, 1), # 4 x 4
            nn.Conv2d(8*convolution_factor, 1, 4, 2, 0), # 1 x 1
            nn.Sigmoid(),
        )
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
    
    def forward(self, X):
        return self.model(X)


class Generator(nn.Module):
    def __init__(self, z_dim, features_g, img_channels):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            # Input: N x z_dim x 1 x 1
            self._block(z_dim, features_g*16, 4, 1, 0), # N x 1024 x 4 x 4
            self._block(features_g*16, features_g*8, 4, 2, 1), # N x 512 x 8 x 8
            self._block(features_g*8, features_g*4, 4, 2, 1), # N x 256 x 16 x 16
            self._block(features_g*4, features_g*2, 4, 2, 1), # N x 128 x 32 x 32
            nn.ConvTranspose2d(features_g*2, img_channels, 4, 2, 1), # N x 3 x 64 x 64
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
