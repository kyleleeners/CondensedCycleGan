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

    def __init__(self):
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

        self.wc = WaveNetClassifier(encoder_dict, 199980)

    def forward(self, x):
        return self.wc.forward(x)


class Generator(nn.Module):

    def __init__(self, dim=128):
        super(Generator, self).__init__()

        conv_bn_relu = conv_norm_act
        dconv_bn_relu = dconv_norm_act

        self.ds = nn.Sequential(conv_bn_relu(2, dim, 5, 5),
                                conv_bn_relu(dim, dim, 3, 3))

        self.res = ResNet1D(dim, dim, dim, 8)

        self.us = nn.Sequential(dconv_bn_relu(dim, dim, 3, 3),
        						dconv_bn_relu(dim, dim, 5, 5),
                                nn.Conv1d(dim, 2, 1, 1, 0, bias=False))
        #nn.ConvTranspose1d(1, 1, 6, 6, 0, 0, bias=False)
        #Add a long scale filter to help with gibbs.
        self.deGibbs = nn.Conv1d(1, 1, 1001, 1, 500, bias=False)

    def forward(self, x):

    	#Fourier Transform the signal
        rfft_squeeze = torch.rfft(x, 2).squeeze(1)
        rfft_squeeze_transpose = torch.transpose(rfft_squeeze, dim0=1, dim1=2)

        #Downsample fft
        down_sample = self.ds(rfft_squeeze_transpose)

        #Pass through resnet
        res_out = self.res.forward(down_sample)

        #Upsample freqs
        up_sample = self.us(res_out)
        #print(up_sample.shape)

        #Inverse fft
        up_sample_transpose_unsqueeze = torch.transpose(up_sample, dim0=1, dim1=2).unsqueeze(1)
        print(up_sample_transpose_unsqueeze.shape)

        ifft = torch.irfft(up_sample_transpose_unsqueeze, 2, signal_sizes=x.shape[1:])

        #Kill the rining with final filter.
        out = self.deGibbs(ifft)

        #Output between [-1,1]
        out = torch.tanh(out)

        return out