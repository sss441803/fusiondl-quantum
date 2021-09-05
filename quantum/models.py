import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as opt
from torch.nn.utils import weight_norm
from .QConv1D import QConv1D, EasyQConv1D, DenseQConv1D, MoreParamDenseQConv1D
import numpy as np

class FTCN(nn.Module):
    def __init__(self, n_scalars, n_profiles, profile_size, channels_spatial, kernel_spatial, linear_sizes, channels_temporal, kernel_temporal, output_size, ancillas=2, dropout=0.1):
        super(FTCN, self).__init__()
        self.inputblock = InputBlock(n_scalars, n_profiles, profile_size, channels_spatial, kernel_spatial, linear_sizes, dropout)
        print('InputBlock parameters: ', n_scalars, n_profiles, profile_size, channels_spatial, kernel_spatial, linear_sizes, dropout)
        self.input_layer = TimeDistributed(self.inputblock, batch_first=True)
        self.tcn_in_size = self.inputblock.out_size
        self.tcn = TCN(self.tcn_in_size, channels_temporal, kernel_temporal, output_size, ancillas=ancillas, dropout=dropout)
        print('TCN parameters: ', self.tcn_in_size, channels_temporal, kernel_temporal, output_size, ancillas, dropout)
        self.model = nn.Sequential(self.input_layer,self.tcn)
    def forward(self,x):
        return self.model(x)

# Dimensions of the input should be (n_batch, n_scalars + n_profiles*profile_size)
class InputBlock(nn.Module):
    def __init__(self, n_scalars, n_profiles, profile_size, channels: list, kernel_size, linear_sizes: list, dropout=0.2) -> nn.Module:
        super(InputBlock, self).__init__()
        self.pooling_size = 2
        self.n_scalars = n_scalars
        self.n_profiles = n_profiles
        self.profile_size = profile_size
        self.conv_output_size = profile_size
        if self.n_profiles == 0:
            self.net = None
            self.conv_output_size = 0
            self.tcn_in_size = n_scalars
        else: # Build spatial convolution layers for 1D profile signals
            self.layers = nn.ModuleList()
            in_channels = n_profiles # Number of input channels for the next convolution layer
            for (i, layer) in enumerate(channels):
                layer_type = layer[0]
                out_channels = int(layer[1:])
                if layer_type == 'e': # Add a quantum layer
                    if i == 0:
                        self.layers.append(EasyQConv1D(in_channels, out_channels, kernel_size, initial=True))
                    else:
                        self.layers.append(EasyQConv1D(in_channels, out_channels, kernel_size))
                    self.layers.append(nn.ReLU())
                    self.conv_output_size = calculate_conv_output_size(self.conv_output_size,0,1,1,kernel_size)
                    self.layers.append(nn.MaxPool1d(kernel_size=self.pooling_size))
                    print('Quantum convolution with channels ', in_channels, out_channels)
                elif layer_type == 'q': # Add a new quantum layer
                    self.layers.append(QConv1D(in_channels, out_channels, kernel_size))
                    self.layers.append(nn.ReLU())
                    self.conv_output_size = calculate_conv_output_size(self.conv_output_size,0,1,1,kernel_size)
                    self.layers.append(nn.MaxPool1d(kernel_size=self.pooling_size))
                    print('New Quantum convolution with channels ', in_channels, out_channels)
                else: # Add a classical layer
                    self.layers.append(nn.Conv1d(in_channels, out_channels, kernel_size))
                    self.layers.append(nn.ReLU())
                    self.conv_output_size = calculate_conv_output_size(self.conv_output_size,0,1,1,kernel_size)
                    self.layers.append(nn.MaxPool1d(kernel_size=self.pooling_size))
                    print('Classical convolution with channels ', in_channels, out_channels)
                self.conv_output_size = calculate_conv_output_size(self.conv_output_size,0,1,self.pooling_size,self.pooling_size)
                self.layers.append(nn.Dropout2d(dropout))
                in_channels = out_channels
            self.convnet = nn.Sequential(*self.layers)
            self.conv_output_size = self.conv_output_size*out_channels

            # Build linear layers
            self.linear_layers = nn.ModuleList()
            linear_size_pre = self.conv_output_size
            for linear_size in linear_sizes:
               self.linear_layers.append(nn.Linear(linear_size_pre,linear_size))
               self.linear_layers.append(nn.ReLU())
               linear_size_pre=linear_size
            self.linear_net = nn.Sequential(*self.linear_layers)

            self.linear_size_final = linear_size
            self.out_size = self.linear_size_final + self.n_scalars
            
    def forward(self, x):
        #print('inputblock x device ', x.device)
        if self.n_profiles == 0:
            full_features = x#x_scalars
        else:
            if self.n_scalars == 0:
                x_profiles = x
            else:
                x_scalars = x[:,:self.n_scalars]
                x_profiles = x[:,self.n_scalars:]
            x_profiles = x_profiles.contiguous().view(x.size(0),self.n_profiles,self.profile_size)
            profile_features = self.convnet(x_profiles).view(x.size(0),-1)
            profile_features = self.linear_net(profile_features)
            if self.n_scalars == 0:
                full_features = profile_features
            else:
                full_features = torch.cat([x_scalars,profile_features],dim=1)

        return full_features



def calculate_conv_output_size(L_in,padding,dilation,stride,kernel_size):
    return int(np.floor((L_in + 2*padding - dilation*(kernel_size-1) - 1)*1.0/stride + 1))


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, layer_type, n_inputs, n_outputs, kernel_size, stride, dilation, padding, ancillas=2, dropout=0.2):
        super(TemporalBlock, self).__init__()
        #self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,stride=stride, padding=padding, dilation=dilation))
        if layer_type == 'q':
            print('Quantum convolution with channels ', n_inputs, n_outputs, ' with kernel size ', kernel_size)
            self.conv1 = QConv1D(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
            self.conv2 = QConv1D(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        elif layer_type == 'e':
            print('Easy quantum convolution with channels ', n_inputs, n_outputs, ' with kernel size ', kernel_size)
            self.conv1 = EasyQConv1D(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
            self.conv2 = EasyQConv1D(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        elif layer_type == 'd':
            print('Dense quantum convolution with channels ', n_inputs, n_outputs, ' with kernel size ', kernel_size, ' and ', ancillas, 'ancilla qubit(s).')
            self.conv1 = DenseQConv1D(n_inputs, n_outputs, kernel_size, ancillas=ancillas, stride=stride, padding=padding, dilation=dilation)
            self.conv2 = DenseQConv1D(n_outputs, n_outputs, kernel_size, ancillas=ancillas, stride=stride, padding=padding, dilation=dilation)
        elif layer_type == 'm':
            print('More Param Dense quantum convolution with channels ', n_inputs, n_outputs, ' with kernel size ', kernel_size)
            self.conv1 = MoreParamDenseQConv1D(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
            self.conv2 = MoreParamDenseQConv1D(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        else:
            self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
            self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout2d(dropout)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout2d(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        #self.init_weights()

    #def init_weights(self):
    #    self.conv1.weight.data.normal_(0, 0.01)
    #    self.conv2.weight.data.normal_(0, 0.01)
    #    if self.downsample is not None:
    #        self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

#dimensions are batch,channels,length
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, channels, kernel_size=2, ancillas=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        self.layers = nn.ModuleList()
        in_channels = num_inputs
        for i, layer in enumerate(channels):
            layer_type = layer[0]
            out_channels = int(layer[1:])
            dilation_size = 2 ** i
            self.layers += [TemporalBlock(layer_type, in_channels, out_channels, kernel_size, ancillas=ancillas, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
            in_channels = out_channels
        self.network = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.network(x)
    
    
class TCN(nn.Module):
    def __init__(self, input_size, channels, kernel_size, output_size, ancillas=2, dropout=0.1):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, channels, kernel_size, ancillas=ancillas, dropout=dropout)
        last_layer = channels[-1]
        self.linear = nn.Linear(int(last_layer[1:]), output_size)
#         self.sig = nn.Sigmoid()

    def forward(self, x):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        output = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        output = self.linear(output)#.transpose(1,2)).transpose(1,2)
        return output
#         return self.sig(output)


class TimeDistributed(nn.Module):
    def __init__(self, module,is_half=False, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first
        self.is_half=is_half

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)
        x=x.float()

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
        if self.is_half:
         y=y.half()
        return y