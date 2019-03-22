import math
import torch

#Causual Convolution This code is borrowed from offical Wavenet code.
#Repo is here: https://github.com/vincentherrmann/pytorch-wavenet
class Conv(torch.nn.Module):
    """
    A convolution with the option to be causal and use xavier initialization
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 dilation=1, bias=True, w_init_gain='linear', is_causal=False):
        super(Conv, self).__init__()
        self.is_causal = is_causal
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    dilation=dilation, bias=bias)

        torch.nn.init.xavier_uniform(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        if self.is_causal:
                padding = (int((self.kernel_size - 1) * (self.dilation)), 0)
                signal = torch.nn.functional.pad(signal, padding) 
        return self.conv(signal)

# Much of this code was recycled from the wavenet github.
#Repo is here: https://github.com/vincentherrmann/pytorch-wavenet
class WaveNet(torch.nn.Module):
    def __init__(self, n_channels, n_cond_channels, n_layers, max_dilation, n_residual_channels, n_skip_channels, n_inputs, upsamp_window, upsamp_stride):

        #Super
        super(WaveNet, self).__init__()

        #Build upsampleing for conditions.
        self.upsample = torch.nn.ConvTranspose1d(n_cond_channels, n_cond_channels, upsamp_window, upsamp_stride)

        #Assign the parameters
        self.n_layers = n_layers
        self.max_dilation = max_dilation
        self.n_residual_channels = n_residual_channels
        self.cond_layers = Conv(n_cond_channels, n_residual_channels*n_layers, w_init_gain='tanh')
        self.n_inputs = n_inputs
        self.n_channels = n_channels
        self.n_cond_channels = n_cond_channels
        self.n_skip_channels = n_skip_channels

        #Make layers
        self.dilate_layers = torch.nn.ModuleList()
        self.res_layers = torch.nn.ModuleList()
        self.skip_layers = torch.nn.ModuleList()


        #Was an embedding but we don't do this anymore.
        self.casualInput = Conv(self.n_channels, self.n_residual_channels, is_causal=True)

        #Final conv layers.
        self.conv_out = Conv(self.n_skip_channels, self.n_channels, bias=False, w_init_gain='relu')
        self.conv_end = Conv(self.n_channels, self.n_channels, bias=False, w_init_gain='linear')

        #Add out final linear layer.
        self.final_linear = torch.nn.Linear(self.n_inputs, 1)

        #Build the loop.
        loop_factor = math.floor(math.log2(max_dilation)) + 1
        for i in range(n_layers):

            #Double dilation up to max dilation.
            dilation = 2**(i % loop_factor)

            #Dilated Conv layer
            d_layer = Conv(n_residual_channels, 2*n_residual_channels, kernel_size=2, dilation=dilation, w_init_gain='tanh', is_causal=True)
            self.dilate_layers.append(d_layer)

            #We don't need a res layer on the last layer.
            if i < n_layers - 1:
                res_layer = Conv(n_residual_channels, n_residual_channels, w_init_gain='linear')
                self.res_layers.append(res_layer)

            #Add the skip layer.
            skip_layer = Conv(n_residual_channels, n_skip_channels, w_init_gain='relu')
            self.skip_layers.append(skip_layer)

    def forward(self, forward_input, features):
        #Upsample features
        cond_input = self.upsample(features)

        #Need to make sure these are the same size.
        assert(cond_input.size(2) >= self.n_inputs)
        if cond_input.size(2) > forward_input.size(1):
            cond_input = cond_input[:, :, :self.n_inputs]

        #Causal Convolution pre-res and skip connections.
        forward_input = self.casualInput(forward_input)

        #Build the conditoning, reshape so we can iterate through number of res-layers.
        cond_acts = self.cond_layers(cond_input)
        cond_acts = cond_acts.view(cond_acts.size(0), self.n_layers, -1, cond_acts.size(2))

        #Res-layer loop
        for i in range(self.n_layers):
            in_act = self.dilate_layers[i](forward_input)

            #Build the gates.
            t_act = torch.tanh(in_act[:, :self.n_residual_channels, :] + cond_acts[:,i,:,:])
            s_act = torch.sigmoid(in_act[:, self.n_residual_channels:, :] + cond_acts[:,i,:,:])
            act = t_act*s_act

            #Apply Res Layer.
            if i < len(self.res_layers):
                res_acts = self.res_layers[i](act)

            #Add res
            forward_input = res_acts + forward_input

            #Add the skip layers.
            if i == 0:
                output = self.skip_layers[i](act)
            else:
                output = self.skip_layers[i](act) + output

        #final output
        output = torch.nn.functional.relu(output, True)
        output = self.conv_out(output)
        output = torch.nn.functional.relu(output, True)
        output = self.conv_end(output)

        #Added on tanh
        output = self.final_linear(output)
        output = torch.tanh(output)

        return output











