import math
import torch
from WaveNetContinuous import WaveNet
import torch.nn as nn
import warnings

#Conv de-conv operations
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

# Use wavenet to generate an output/
class WaveNetGenerator(torch.nn.Module):
	def __init__(self, WaveNetDict, n_past_values, downsample_factor=2):

		#Super
		super(WaveNetGenerator, self).__init__()

		self.downsample_factor = downsample_factor

		#Build WaveNet
		#n_channels, n_cond_channels, n_layers, max_dilation, n_residual_channels, n_skip_channels, n_inputs, upsamp_window, upsamp_stride
		self.wavenet = WaveNet(n_channels=WaveNetDict["n_channels"], n_cond_channels=WaveNetDict["n_channels"], n_layers = WaveNetDict["n_layers"],
		max_dilation = WaveNetDict["max_dilation"], n_residual_channels=WaveNetDict["n_residual_channels"], n_skip_channels = WaveNetDict["n_skip_channels"],
		n_inputs = n_past_values, upsamp_window = 1, upsamp_stride = 1)

		#Build N past values.
		self.n_past_values = n_past_values

		#Add learnable downsample layers.
		self.downsample_layers = torch.nn.ModuleList()
		for i in range(downsample_factor):
			self.downsample_layers.append(conv1d_norm_act(WaveNetDict["n_channels"], WaveNetDict["n_channels"],2,2))

		#Add upsample layers.
		self.upsample_layers = torch.nn.ModuleList()
		for i in range(downsample_factor):
			self.upsample_layers.append(dconv1d_norm_act(WaveNetDict["n_channels"],WaveNetDict["n_channels"],2,2))


	def forward(self, song):

		if song.size(2) % 2**self.downsample_factor != 0:
			warnings.warn("Gong length not divisable by down sample factor.\nOutputs dimensions will not line up.")


		#Downsample.
		for downsample in self.downsample_layers:
			song = downsample(song)

		#concat the zeros with the song vector.
		batch_size = song.size(0)
		num_channels = song.size(1)
		song_length = song.size(2)

		# song is conditional
		conditional = torch.zeros(batch_size, num_channels, self.n_past_values)
		conditional = torch.cat((conditional, song), dim=2)

		# output is initated to zeros.
		outputs = torch.zeros(batch_size,num_channels, song_length + self.n_past_values - 1)

		#run the loop.
		for i in range(song_length):
			#Conditional

			#Not the last point
			if i < song_length - 1:
				outputs[:,:,self.n_past_values+i:self.n_past_values+i+1] = self.wavenet(outputs[:,:,i:(i+self.n_past_values)], conditional[:,:,(i+1):(i+self.n_past_values+1)])
			else:
				#Last point
				outputs[:,:,self.n_past_values+i:self.n_past_values+i+1] = self.wavenet(outputs[:,:,i:(i+self.n_past_values)], conditional[:,:,i:(i+self.n_past_values)])

		#Don't include the 0's
		outputs = outputs[:,:,self.n_past_values-1:]

		#Upsample.
		for upsample in self.upsample_layers:
			outputs = upsample(outputs)

		return outputs