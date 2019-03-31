import torch
from WaveNet2.WaveNetEncoder.WaveNetEncoder import WaveNetEncoder
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

class WaveNetClassifier(torch.nn.Module):
    def __init__(self, encoder_dict, inputsize, activation=torch.nn.Sigmoid()):
        # Super
        super(WaveNetClassifier, self).__init__()

        # Build the Encoder - wrapping the encoder here.
        self.encoder = WaveNetEncoder(n_channels=encoder_dict["n_channels"], n_layers=encoder_dict["n_layers"],
                                      max_dilation=encoder_dict["max_dilation"],
                                      n_residual_channels=encoder_dict["n_residual_channels"],
                                      n_dilated_channels=encoder_dict["n_dilated_channels"],
                                      encoding_factor=encoder_dict["encoding_factor"],
                                      encoding_stride=encoder_dict["encoding_stride"])

        # Figure out encoding size.
        self.ds = conv1d_norm_act(encoder_dict["n_channels"], encoder_dict["n_channels"],
         encoder_dict['down_sample'], encoder_dict['down_sample'])
        zerovect = torch.zeros(1, encoder_dict["n_channels"], inputsize)
        out = self.encoder(self.ds(zerovect))
        self.encodingSize = out.size(2)

        # Build the linear layer.
        self.linear = torch.nn.Linear(self.encodingSize, 1)
        self.activation = activation

    def forward(self, signal):
        # Encode and get Negative log likelyhood.
        signal = self.ds(signal)
        signal = self.encoder(signal)
        signal = self.linear(signal)

        # return p.
        #return self.activation(signal)
        return signal
