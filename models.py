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
            'n_layers':7,
            'max_dilation': 256,
            'down_sample': 6,
            'n_residual_channels': 64,
            'n_dilated_channels': 64,
            'encoding_factor': 250,
            'encoding_stride': 250
        }

        self.wc = WaveNetClassifier(encoder_dict, 399996)

    def forward(self, x):
        return self.wc.forward(x)

class DiscriminatorR(nn.Module):

    def __init__(self, dim=16, n_length=399996, enc_fact=100, ds_fact=4):
        super(DiscriminatorR, self).__init__()

        self.ds = nn.Sequential(conv_bn_relu(1, 1, ds_fact, ds_fact),
                                conv_bn_relu(1, 1, ds_fact, ds_fact))

        self.res = ResNet1D(dim, dim, dim, 4)
        self.conv1=conv_norm_act(1, dim, 101, 50)
        self.conv2=conv_norm_act(dim, 1, enc_fact, enc_fact)

        zeros = torch.zeros(1,n_length).unsqueeze(0)
        zeros = self.ds(zeros)
        zeros = self.conv1(zeros)
        zeros = self.res.forward(zeros)
        zeros = self.conv2(zeros)

        #Test the layers
        self.linear = torch.nn.Linear(zeros.size(2),1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.res.forward(x)
        x = self.conv2(x)
        return self.linear(x)



class Generator(nn.Module):

    def __init__(self, dim=256):
        super(Generator, self).__init__()

        conv_bn_relu = conv_norm_act
        dconv_bn_relu = dconv_norm_act

        self.ds = nn.Sequential(conv_bn_relu(1, 1, 6, 6),
                                conv_bn_relu(1, 1, 6, 6))

        self.res = ResNet1D(2, 2, dim, 6)

        self.us = nn.Sequential(dconv_bn_relu(1, 1, 6, 6),
                                nn.ConvTranspose1d(1, 1, 6, 6, 0, 0, bias=False))
        #nn.ConvTranspose1d(1, 1, 6, 6, 0, 0, bias=False)
        #Add a long scale filter to help with gibbs.
        self.deGibbs = nn.Conv1d(1, 1, 1001, 1, 500, bias=False)

    def forward(self, x):
        down_sample = self.ds(x)

        rfft_squeeze = torch.rfft(down_sample, 2).squeeze(1)
        rfft_squeeze_transpose = torch.transpose(rfft_squeeze, dim0=1, dim1=2)

        res_out = self.res.forward(rfft_squeeze_transpose)

        res_out_transpose_unsqueeze = torch.transpose(res_out, dim0=1, dim1=2).unsqueeze(1)
        ifft = torch.irfft(res_out_transpose_unsqueeze, 2, signal_sizes=down_sample.shape[1:])

        up_sample = self.us(ifft)

        #Hopefully this can learn to kill ringing.
        out = self.deGibbs(up_sample)

        out = torch.tanh(out)

        return out