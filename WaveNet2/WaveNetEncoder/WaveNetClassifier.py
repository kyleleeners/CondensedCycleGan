import math
import torch
from WaveNetEncoder import WaveNetEncoder

class WaveNetClassifier(torch.nn.Module):
	def __init__(self, EncoderDict, inputsize, activation=torch.nn.Sigmoid()):

		#Super
		super(WaveNetClassifier, self).__init__()

		#Build the Encoder - wrapping the encoder here.
		self.encoder = WaveNetEncoder(n_channels=EncoderDict["n_channels"], n_layers=EncoderDict["n_layers"], max_dilation=EncoderDict["max_dilation"],
		 n_residual_channels=EncoderDict["n_residual_channels"], n_dilated_channels=EncoderDict["n_dilated_channels"],
		  encoding_factor=EncoderDict["encoding_factor"], encoding_stride=EncoderDict["encoding_stride"])

		#Figure out encoding size.
		zerovect = torch.zeros(1, EncoderDict["n_channels"], inputsize)
		out = self.encoder(zerovect)
		self.encodingSize = out.size(2)

		#Build the linear layer.
		self.linear = torch.nn.Linear(self.encodingSize,1)
		self.activation = activation


	def forward(self, signal):
		#Encode and get Negative log likelyhood.
		signal = self.encoder(signal)
		signal = self.linear(signal)

		#return p.
		return self.activation(signal)








