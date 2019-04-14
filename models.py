from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch

from ResNet.ResNet1d import ResNet1D
from WaveNet2.WaveNetEncoder.WaveNetClassifier import WaveNetClassifier
#from WaveNet2.WaveNetGenerator import WaveNetGenerator


def conv_norm_act(in_dim, out_dim, kernel_size, stride, padding=0,
                  norm=nn.BatchNorm1d, act=nn.Tanh):
    return nn.Sequential(
        nn.Conv1d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        norm(out_dim),
        act())


def dconv_norm_act(in_dim, out_dim, kernel_size, stride, padding=0,
                   output_padding=0, norm=nn.BatchNorm1d, act=nn.Tanh):
    return nn.Sequential(
        nn.ConvTranspose1d(in_dim, out_dim, kernel_size, stride,
                           padding, output_padding, bias=False),
        norm(out_dim),
        act())


class Discriminator(nn.Module):

    def __init__(self, dim=128):
        super(Discriminator, self).__init__()

        encoder_dict = {
            'n_channels': 1,
            'n_layers':8,
            'max_dilation': 512,
            'down_sample': 14,
            'n_residual_channels': 64,
            'n_dilated_channels': 64,
            'encoding_factor': 250,
            'encoding_stride': 250
        }

        self.wc = WaveNetClassifier(encoder_dict, 149997)

    def forward(self, x):
        return self.wc.forward(x)


class Generator(nn.Module):

    def __init__(self, dim=128):
        super(Generator, self).__init__()

        conv_bn_relu = conv_norm_act
        dconv_bn_relu = dconv_norm_act

        self.ds = nn.Sequential(conv_bn_relu(1, 128, 15, 2, 1),
                                conv_bn_relu(128, 256, 5, 2, 2),
                                conv_bn_relu(256, 1, 5, 2, 2))

        self.res = ResNet1D(1, 1, 512, 9)

        self.us = nn.Sequential(dconv_bn_relu(1, 256, 5, 2, 2),
                                dconv_bn_relu(256, 128, 5, 2, 2),
                                dconv_bn_relu(128, 1, 15, 2, 1))

        #nn.ConvTranspose1d(1, 1, 6, 6, 0, 0, bias=False)
        #Add a long scale filter to help with gibbs.
        self.deGibbs = nn.Conv1d(1, 1, 1001, 1, 500, bias=False)

    def forward(self, x):
        down_sample = self.ds(x)

        res_out = self.res.forward(down_sample)

        up_sample = self.us(res_out)

        return up_sample