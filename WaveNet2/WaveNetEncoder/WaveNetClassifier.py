import torch
from WaveNet2.WaveNetEncoder.WaveNetEncoder import WaveNetEncoder


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
        zerovect = torch.zeros(1, encoder_dict["n_channels"], inputsize)
        out = self.encoder(zerovect)
        self.encodingSize = out.size(2)

        # Build the linear layer.
        self.linear = torch.nn.Linear(self.encodingSize, 1)
        self.activation = activation

    def forward(self, signal):
        # Encode and get Negative log likelyhood.
        signal = self.encoder(signal)
        signal = self.linear(signal)

        # return p.
        #return self.activation(signal)
        return signal
