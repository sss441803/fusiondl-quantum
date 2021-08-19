import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as opt
from torch.nn.utils import weight_norm
from convlstmnet import *
import numpy as np

class FLSTM(nn.Module):
  def __init__(self,input_dim=14,output_dim=1,rnn_size=200,dense_size=8,dropout=0.1,batch_first=True,bidirectional=False,rnn_layers=2,n_scalars=14,n_profiles=2,profile_size=64,layer_sizes_spatial=[16,8,8], kernel_size_spatial=3, linear_size=5, linear_layer_num=2,device=None): 
      super(FLSTM, self).__init__()
      self.n_profiles = n_profiles
      self.pre_rnn_network = InputBlock(n_scalars, n_profiles,profile_size, layer_sizes_spatial, kernel_size_spatial, linear_size, dropout,linear_layer_num)
      self.input_layer = TimeDistributed(self.pre_rnn_network,batch_first=True)
      self.input_dim=self.pre_rnn_network.tcn_in_size
      self.rnn_size=rnn_size
      self.dropout=dropout
      self.rnn_layers=rnn_layers
      self.output_dim=output_dim
      if device==None:
         self.device=torch.device('cuda')
      else: 
         self.device=device
      self.rnn=nn.LSTM(self.input_dim,self.rnn_size,batch_first=True,num_layers=self.rnn_layers).to(self.device)
      self.dropout_layer=nn.Dropout(p=self.dropout)
      self.final_linear=nn.Linear(self.rnn_size,self.output_dim).to(self.device)
  def forward(self, x):
        if self.n_profiles>0:
          x = self.input_layer(x)
        y, _ = self.rnn(x)
        x = y
        x = self.dropout_layer(x)
        x = self.final_linear(x)
        return x




class FTTLSTM(nn.Module):
  def __init__(self,tt_dense=20,cell_steps=10,cell_rank=5,cell_order=1,input_dim=14,output_dim=1,dense_size=8,dropout=0.1,batch_first=True,n_scalars=14,n_profiles=2,profile_size=64,layer_sizes_spatial=[16,8,8], kernel_size_spatial=3,  linear_layer_num=2,device=None): 
      super(FTTLSTM, self).__init__()
      self.n_profiles = n_profiles
      linear_size = dense_size
      self.pre_rnn_network = InputBlock(n_scalars, n_profiles,profile_size, layer_sizes_spatial, kernel_size_spatial, linear_size, dropout,linear_layer_num)
      self.input_layer = TimeDistributed(self.pre_rnn_network,batch_first=True)
      self.input_dim = self.pre_rnn_network.tcn_in_size
      if device == None:
         self.device = torch.device('cuda')
      else: 
         self.device = device
      self.rnn = ConvLSTMNet(
        input_channels = self.input_dim,
        layers_per_block = (1,1),
        hidden_channels = (tt_dense, tt_dense),
        skip_stride = None,
        cell = 'convttlstm', cell_params = {"order": cell_order,
        "steps": cell_steps, "rank": cell_rank },
        kernel_size = 1, dropout=dropout, bias = True,
        output_sigmoid = True)
  def forward(self, x):
        if self.n_profiles>0:
          x = self.input_layer(x)
          y = self.rnn(x)
        return y








class FTCN(nn.Module):
    def __init__(self,n_scalars,n_profiles,profile_size,layer_sizes_spatial,
                 kernel_size_spatial,linear_size,output_size,
                 num_channels_tcn,kernel_size_temporal,dropout=0.1,linear_layer_num=2):
        super(FTCN, self).__init__()
        self.lin = InputBlock(n_scalars, n_profiles,profile_size, layer_sizes_spatial, kernel_size_spatial, linear_size, dropout,linear_layer_num)
        self.input_layer = TimeDistributed(self.lin,batch_first=True)
        self.tcn = TCN(self.lin.tcn_in_size, output_size, num_channels_tcn , kernel_size_temporal, dropout)
        self.model = nn.Sequential(self.input_layer,self.tcn)
    
    def forward(self,x):
        return self.model(x)
    
    def to(self, *args, **kwargs):
        print('FTCN to device')
        self = super().to(*args, **kwargs)
        for _, layer in enumerate(self.model):
            layer = layer.to(*args, **kwargs)
        return self



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
            layer_size_i=1
            for (i,layer_size) in enumerate(layer_sizes):
                layer_size_i=layer_size#//2**i
                if layer_size_i<1:
                   layer_size_i=1
                if i == 0:
                    input_size = n_profiles
                else:
                    input_size = layer_sizes[i-1]
                self.layers.append(weight_norm(nn.Conv1d(input_size, layer_size_i, kernel_size)))
                self.layers.append(nn.ReLU())
                self.conv_output_size = calculate_conv_output_size(self.conv_output_size,0,1,1,kernel_size)
                self.layers.append(nn.MaxPool1d(kernel_size=self.pooling_size))
                self.conv_output_size = calculate_conv_output_size(self.conv_output_size,0,1,self.pooling_size,self.pooling_size)
                self.layers.append(nn.Dropout2d(dropout))
            self.layers = nn.ModuleList(self.layers)
            self.net = nn.Sequential(*self.layers)
            self.conv_output_size = self.conv_output_size*layer_size_i
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

#     def init_weights(self):
#         self.conv1.weight.data.normal_(0, 0.01)
#         self.conv2.weight.data.normal_(0, 0.01)
#         if self.downsample is not None:
#             self.downsample.weight.data.normal_(0, 0.01)

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

    def to(self, *args, **kwargs):
        print('Inputblock to device')
        self = super().to(*args, **kwargs) 
        for _, layer in enumerate(self.net):
            layer = layer.to(*args, **kwargs)
        for _, layer in enumerate(self.linear_net):
            layer = layer.to(*args, **kwargs)
        return self




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
    def to(self, *args, **kwargs):
        print('TimeDistributed to device')
        self = super().to(*args, **kwargs) 
        self.module = self.module.to(*args, **kwargs) 
        return self



