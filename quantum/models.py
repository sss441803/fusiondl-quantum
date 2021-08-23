import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as opt
from torch.nn.utils import weight_norm
from .QConv1D import QConv1D
import numpy as np

class FTCN(nn.Module):
    def __init__(self,n_scalars,n_profiles,profile_size,layer_sizes_spatial,
                 kernel_size_spatial,linear_size,output_size,
                 num_channels_tcn,kernel_size_temporal,dropout=0.1,linear_layer_num=2):
        super(FTCN, self).__init__()
        self.lin = InputBlock(n_scalars, n_profiles,profile_size, layer_sizes_spatial, kernel_size_spatial, linear_size, dropout,linear_layer_num)
        print('InputBlock parameters: ', n_scalars, n_profiles,profile_size, layer_sizes_spatial, kernel_size_spatial, linear_size, dropout,linear_layer_num)
        self.input_layer = TimeDistributed(self.lin,batch_first=True)
        self.tcn = TCN(self.lin.tcn_in_size, output_size, num_channels_tcn, kernel_size_temporal, dropout)
        print('TCN parameters: ', self.lin.tcn_in_size, output_size, num_channels_tcn, kernel_size_temporal, dropout)
        self.model = nn.Sequential(self.input_layer,self.tcn)
    def forward(self,x):
        return self.model(x)

class InputBlock(nn.Module):
    def __init__(self, n_scalars, n_profiles,profile_size, layer_sizes, kernel_size, linear_size, dropout=0.2,linear_layer_num=2):
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
        else:
            self.layers = []
            previous_layer_size = n_profiles
            for (i,layer_size) in enumerate(layer_sizes):
                layer_type = layer_size[0]
                layer_size = int(layer_size[1:])
                if layer_type == 'q':
                    if i == 0:
                        self.layers.append(QConv1D(previous_layer_size, layer_size, kernel_size, initial=True))
                    else:
                        self.layers.append(QConv1D(previous_layer_size, layer_size, kernel_size))
                    self.conv_output_size = calculate_conv_output_size(self.conv_output_size,0,1,1,kernel_size)
                    self.layers.append(nn.MaxPool1d(kernel_size=self.pooling_size))
                    print('Quantum convolution with channels ', previous_layer_size, layer_size)
                else:
                    self.layers.append(nn.Conv1d(previous_layer_size, layer_size, kernel_size))
                    self.layers.append(nn.ReLU())
                    self.conv_output_size = calculate_conv_output_size(self.conv_output_size,0,1,1,kernel_size)
                    self.layers.append(nn.MaxPool1d(kernel_size=self.pooling_size))
                    print('Classical convolution with channels ', previous_layer_size, layer_size)
                self.conv_output_size = calculate_conv_output_size(self.conv_output_size,0,1,self.pooling_size,self.pooling_size)
                self.layers.append(nn.Dropout2d(dropout))
                previous_layer_size = layer_size
            self.layers = nn.ModuleList(self.layers)
            self.net = nn.Sequential(*self.layers)
            self.conv_output_size = self.conv_output_size*layer_size
            self.linear_layers = []
        
         # print("Final feature size = {}".format(self.n_scalars + self.conv_output_size))
            self.linear_layers.append(nn.Linear(self.conv_output_size,linear_size))
            self.linear_layers.append(nn.ReLU())
            linear_size_ll_pre=linear_size
            linear_size_ll=linear_size
            for ll in range(1,linear_layer_num):
               linear_size_ll=linear_size_ll_pre//4
               if linear_size_ll<3:
                  linear_size_ll=3
               self.linear_layers.append(nn.Linear(linear_size_ll_pre,linear_size_ll))
               self.linear_layers.append(nn.ReLU())
               linear_size_ll_pre=linear_size_ll
            self.linear_size_f=linear_size_ll
            self.tcn_in_size=self.linear_size_f+self.n_scalars
            self.linear_layers = nn.ModuleList(self.linear_layers)
            self.linear_net = nn.Sequential(*self.linear_layers)

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
            #print('inputblock x_profiles device ', x_profiles.device)
            x_profiles = x_profiles.contiguous().view(x.size(0),self.n_profiles,self.profile_size)
            #print('inputblock x_profiles device ', x_profiles.device)
            profile_features = self.net(x_profiles).view(x.size(0),-1)
            #print('inputblock profile_features device ', profile_features.device)
            profile_features = self.linear_net(profile_features)
            if self.n_scalars == 0:
                full_features = profile_features
            else:
                full_features = torch.cat([x_scalars,profile_features],dim=1)
#         out = self.net(x)
#         res = x if self.downsample is None else self.downsample(x)
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
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        #self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,stride=stride, padding=padding, dilation=dilation))
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout2d(dropout)

        #self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout2d(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

#dimensions are batch,channels,length
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        self.layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            self.layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        
        self.layers = nn.ModuleList(self.layers)
        self.network = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.network(x)
    
    
class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
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