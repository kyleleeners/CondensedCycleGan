from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch

from ResNet.ResNet1d import ResNet1D
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

        conv_bn_relu = conv_norm_act

        self.ds = nn.Sequential(conv_bn_relu(1, 64, 15, 2, 1),
                                conv_bn_relu(64, 128, 5, 2, 2),
                                conv_bn_relu(128, 256, 5, 2, 2))

        self.fc = nn.Linear(18749,1)

    def forward(self, x):

        ds = self.ds(x)
        fc = self.fc(ds)

        return torch.sigmoid(fc)


class Generator(nn.Module):

    def __init__(self, dim=128):
        super(Generator, self).__init__()

        conv_bn_relu = conv_norm_act
        dconv_bn_relu = dconv_norm_act

        self.ds = nn.Sequential(conv_bn_relu(1, 128, 15, 2, 1),
                                conv_bn_relu(128, 256, 5, 2, 2),
                                conv_bn_relu(256, 1, 5, 2, 2))

        self.res = ResNet1D(2, 2, 512, 9)

        self.us = nn.Sequential(dconv_bn_relu(1, 256, 5, 2, 2),
                                dconv_bn_relu(256, 128, 5, 2, 2),
                                dconv_bn_relu(128, 1, 15, 2, 1))

        #nn.ConvTranspose1d(1, 1, 6, 6, 0, 0, bias=False)
        #Add a long scale filter to help with gibbs.
        self.deGibbs = nn.Conv1d(1, 1, 1001, 1, 500, bias=False)

    def forward(self, x):
        down_sample = self.ds(x)

        fft = torch.transpose(torch.rfft(down_sample, 2).squeeze(1), dim0=1, dim1=2)

        res_out = self.res.forward(fft)

        ifft = torch.irfft(torch.transpose(res_out, dim0=1, dim1=2).unsqueeze(1), 2, signal_sizes=down_sample.shape[1:])

        up_sample = self.us(ifft)

        #Hopefully this can learn to kill ringing.
        out = self.deGibbs(up_sample)

        out = torch.tanh(out)

        return out