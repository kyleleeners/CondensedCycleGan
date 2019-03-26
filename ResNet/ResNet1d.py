import torch
import torch.nn as nn

#1D Convulution use from wavenet code here: TODO: put link here
def conv1d_norm_act(in_dim, out_dim, kernel_size, stride, padding=0,
				  norm=nn.BatchNorm1d, relu=nn.ReLU):
	return nn.Sequential(
		nn.Conv1d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
		norm(out_dim),
		relu())


def dconv1d_norm_act(in_dim, out_dim, kernel_size, stride, padding=0,
				   output_padding=0, norm=nn.BatchNorm1d, relu=nn.ReLU):
	return nn.Sequential(
		nn.ConvTranspose1d(in_dim, out_dim, kernel_size, stride,
						   padding, output_padding, bias=False),
		norm(out_dim),
		relu())

class ResiduleBlock(nn.Module):

	def __init__(self, in_dim, out_dim):
		super(ResiduleBlock, self).__init__()

		conv_bn_relu = conv1d_norm_act

		self.ls = nn.Sequential(conv_bn_relu(in_dim, out_dim, 3, 1, 1),
								nn.Conv1d(out_dim, out_dim, 3, 1, 1),
								nn.BatchNorm1d(out_dim))

	def forward(self, x):
		return x + self.ls(x)

#ResNet code following main paper: https://arxiv.org/pdf/1512.03385.pdf
class ResNet1D(torch.nn.Module):
	def __init__(self, n_channel_in, n_channel_out, r_channels, n_length):

		#Super
		super(ResNet1D, self).__init__()

		conv_bn_relu = conv1d_norm_act

		#Build the Net
		self.n_length = n_length

		#Blocks list.
		self.blocks = nn.ModuleList()

		#first block
		self.blocks.append(conv_bn_relu(n_channel_in, r_channels, 3, 1, 1))

		#middle blocks
		for i in range(n_length-2):
			self.blocks.append(ResiduleBlock(r_channels, r_channels))

		#last block.
		#self.blocks.append(conv_bn_relu(r_channels, n_channel_out, 3, 1, 1))
		self.blocks.append(nn.Conv1d(r_channels, n_channel_out, 3, 1, 1))

	#Forward model.
	def forward(self, signal):
		#Run the forward model.
		for block in self.blocks:
			signal = block(signal)

		return signal








