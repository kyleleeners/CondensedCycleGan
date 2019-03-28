from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch

from ResNet.ResNet1d import ResNet1D
from WaveNet2.WaveNetEncoder.WaveNetClassifier import WaveNetClassifier


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

    def __init__(self):
        super(Discriminator, self).__init__()

        encoder_dict = {
            'n_channels': 1,
            'n_layers': 5,
            'max_dilation': 1024,
            'n_residual_channels': 8,
            'n_dilated_channels': 8,
            'encoding_factor': 500,
            'encoding_stride': 500
        }

        self.wc = WaveNetClassifier(encoder_dict, 399996)

    def forward(self, x):
        return self.wc.forward(x)



class Generator(nn.Module):

    def __init__(self, dim=32):
        super(Generator, self).__init__()

        conv_bn_relu = conv_norm_act
        dconv_bn_relu = dconv_norm_act

        self.ds = nn.Sequential(conv_bn_relu(1, 1, 3, 3),
                                conv_bn_relu(1, 1, 3, 3))

        self.res = ResNet1D(2, 2, dim, 6)

        self.us = nn.Sequential(dconv_bn_relu(1, 1, 3, 3),
                                dconv_bn_relu(1, 1, 3, 3))

        #Add a long scale filter to help with gibbs.
        self.deGibbs = nn.Conv1d(1, 1, 1001, 1, 500, bias=False)

    def forward(self, x):
        down_sample = self.ds(x)

        fft = torch.rfft(down_sample, 3)
        fft_in = torch.transpose(fft, dim0=2, dim1=3).squeeze(1)

        res_out = self.res.forward(fft_in)

        fft_out = torch.transpose(res_out, dim0=1, dim1=2).unsqueeze(1)
        ifft = torch.irfft(fft_out, 2, signal_sizes=down_sample.shape[1:])

        up_sample = self.us(ifft)

        #Hopefully this can learn to kill ringing.
        out = self.deGibbs(up_sample)

        out = torch.tanh(out)

        return out
