import torch
import torch.nn as nn

import time

from .QConv_Kernel import ConvKernel, entangle_mat
from .QConv_Kernel_copy import ConvKernel as newConvKernel

# Need decompose a multichannel quantum convolution layer into many single channel quantum convolution. This is because each torch.nn.Module converted from a qml.qnode must be associated with a device that contains all the needed qubits. Having all the qubits in one layer is too many for one device.

# This kernel is for classical CNN and can be used inplace of the quantum kernel

# Initialize convolutional layer with the custom kernel
class CustomKernel_1In1Out_Conv1D(nn.Module):
    def __init__(self, kernel_module: nn.Module, kernel_size, stride=1): #(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        super(CustomKernel_1In1Out_Conv1D, self).__init__()
        self.kernel_module = kernel_module
        self.kernel_size = kernel_size
        self.stride = stride
    def memory_strided_im2col(self, x, kernel_size, stride):
        # x has dimension (n_batch, n_channels, im_width, im_height)
        output_shape = x.shape[-1] - kernel_size + 1
        out = x.unfold(-1,kernel_size,stride)
        #print('shape after unfold: ', out.shape)
        out = out.reshape(x.shape[0], output_shape, kernel_size)
        #print('shape after reshape: ', out.shape)
        return out
    def forward(self, x):
        output_shape = x.shape[-1] - self.kernel_size + 1
        n_batch = x.size(0)
        x = self.memory_strided_im2col(x, self.kernel_size, self.stride)
        #print('shape after memory stride ', x.shape)
        x = x.reshape(-1, self.kernel_size)
        #print('shape after reshape ', x.shape)
        out = self.kernel_module(x)
        #print('shape after kernel ', out.shape)
        out = out.reshape(n_batch, -1, output_shape)
        #print('out shape ', out.shape)
        return out
    def to(self, *args, **kwargs):
        print('Q_MulIn1Out_Conv1D to device')
        self = super().to(*args, **kwargs)
        self.kernel_module = self.kernel_module.to(*args, **kwargs)
        return self

# 1In1Out channel quantum convolution 2D
class Q_1In1Out_Conv1D(nn.Module):
    def __init__(self, entangle_matrix, kernel_size, stride=(1,1)):
        super(Q_1In1Out_Conv1D, self).__init__()
        q_kernel = ConvKernel(entangle_matrix, kernel_size)
        self.conv = CustomKernel_1In1Out_Conv1D(q_kernel, kernel_size, stride=stride)
    def forward(self, x):
        out = self.conv(x)
        return out
    def to(self, *args, **kwargs):
        print('CustomKernel_1In1Out_Conv1D to device')
        self = super().to(*args, **kwargs)
        self.conv = self.conv.to(*args, **kwargs)
        return self

# Multichannel input one channel output quantum convolution 2D
class Separate_Q_MulIn1Out_Conv1D(nn.Module):
    def __init__(self, entangle_matrix, in_channels, kernel_size, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.linear = nn.Linear(in_channels, 1)
        self.convs = []
        for _ in range(self.in_channels):
            self.convs.append(Q_1In1Out_Conv1D(entangle_matrix, kernel_size, stride))
        self.convs = nn.ModuleList(self.convs)
    def forward(self, x):
        device = x.device
        output = torch.tensor([]).to(device)
        for channel in range(self.in_channels):
            x_channel = x[:, channel]
            conv = self.convs[channel]
            out = conv(x_channel)
            out = out.unsqueeze(1)
            output = torch.cat((output, out), dim=1)
            #print('in_channel: ', channel)
        output = self.linear(torch.transpose(output, 1, -1))
        output = torch.transpose(output, 1, -1)
        #print('output shape', output.shape)
        return output
    def to(self, *args, **kwargs):
        print('Q_MulIn1Out_Conv1D to device')
        self = super().to(*args, **kwargs)
        self.linear = self.linear.to(*args, **kwargs)
        for channel in range(self.in_channels):
            self.convs[channel].to(*args, **kwargs)
        return self

# Quantum convolution 2D layer
class SeparateQConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.out_channels = out_channels
        n_qubits = kernel_size
        ngpus = torch.cuda.device_count()
        self.entangle_matrix = entangle_mat(n_qubits) if ngpus == 0 else [entangle_mat(n_qubits).to('cuda:'+str(gpu)) for gpu in range(ngpus)]
        self.convs = []
        for _ in range(self.out_channels):
            self.convs.append(Separate_Q_MulIn1Out_Conv1D(self.entangle_matrix, in_channels, kernel_size, stride))
        self.convs = nn.ModuleList(self.convs)
    def forward(self, x):
        device = x.device
        output = torch.tensor([]).to(device)
        for channel in range(self.out_channels):
            conv = self.convs[channel]
            out = conv(x)
            output = torch.cat((output, out), dim=1)
            #print('out_channel: ', channel)
        output = output.squeeze(-2)
        #print('Convolution output shape', output.shape)
        return output
    def to(self, *args, **kwargs):
        print('QConv1D to device')
        self = super().to(*args, **kwargs) 
        for channel in range(self.out_channels):
            self.convs[channel].to(*args, **kwargs)
        return self

# Multichannel input one channel output quantum convolution 2D
class Q_MulIn1Out_Conv1D(nn.Module):
    def __init__(self, entangle_matrix, in_channels, kernel_size, stride=1, initial=False):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.q_kernel = ConvKernel(entangle_matrix, kernel_size*in_channels, initial=initial)
        #self.weight = self.q_kernel.weight
    def memory_strided_im2col(self, x, in_channels, kernel_size, stride):
        # x has dimension (n_batch, n_channels, length)
        output_shape = x.shape[-1] - kernel_size + 1
        out = x.unfold(-1,kernel_size,stride)
        out = torch.cat([out[:,i] for i in range(out.shape[1])], -1)
        return out
    def forward(self, x):
        output_shape = x.shape[-1] - self.kernel_size + 1
        n_batch = x.size(0)
        x = self.memory_strided_im2col(x, self.in_channels, self.kernel_size, self.stride)
        #print('memory strided im2col succeed')
        x = x.reshape(-1, self.kernel_size*self.in_channels)
        output = self.q_kernel(x)
        #print('q_kernel successful with output shape ', output.shape)
        output = output.reshape(n_batch, 1, -1, output_shape)
        #print('output shape ', output.shape)
        return output

# Quantum convolution 2D layer
class QConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, initial=False):
        super().__init__()
        print('channels: ', in_channels, out_channels)
        self.out_channels = out_channels
        n_qubits = kernel_size
        ngpus = torch.cuda.device_count()
        self.entangle_matrix = entangle_mat(n_qubits*in_channels) if ngpus == 0 else [entangle_mat(n_qubits*in_channels).to('cuda:'+str(gpu)) for gpu in range(ngpus)]
        self.convs = []
        for _ in range(self.out_channels):
            self.convs.append(Q_MulIn1Out_Conv1D(self.entangle_matrix, in_channels, kernel_size, stride=stride, initial=initial))
        self.convs = nn.ModuleList(self.convs)
        #self.weight = nn.Parameter([conv.weight for conv in self.convs])
    def forward(self, x):
        device = x.device
        output = torch.tensor([]).to(device)
        for channel in range(self.out_channels):
            conv = self.convs[channel]
            out = conv(x)
            output = torch.cat((output, out), dim=1)
            #print('out_channel: ', channel)
        output = output.squeeze(-2)
        #print('Convolution output shape', output.shape)
        return output

# Multichannel input one channel output quantum convolution 2D
class New_Q_MulIn1Out_Conv1D(nn.Module):
    def __init__(self, entangle_matrix, in_channels, kernel_size, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.q_kernel = newConvKernel(entangle_matrix, kernel_size*in_channels)
        #self.weight = self.q_kernel.weight
    def memory_strided_im2col(self, x, in_channels, kernel_size, stride):
        # x has dimension (n_batch, n_channels, length)
        output_shape = x.shape[-1] - kernel_size + 1
        out = x.unfold(-1,kernel_size,stride)
        out = torch.cat([out[:,i] for i in range(out.shape[1])], -1)
        return out
    def forward(self, x):
        output_shape = x.shape[-1] - self.kernel_size + 1
        n_batch = x.size(0)
        x = self.memory_strided_im2col(x, self.in_channels, self.kernel_size, self.stride)
        #print('memory strided im2col succeed')
        x = x.reshape(-1, self.kernel_size*self.in_channels)
        output = self.q_kernel(x)
        #print('q_kernel successful with output shape ', output.shape)
        output = output.reshape(n_batch, 1, -1, output_shape)
        #print('output shape ', output.shape)
        return output

# Quantum convolution 2D layer
class newQConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        print('channels: ', in_channels, out_channels)
        self.out_channels = out_channels
        n_qubits = kernel_size
        ngpus = torch.cuda.device_count()
        self.entangle_matrix = entangle_mat(n_qubits*in_channels) if ngpus == 0 else [entangle_mat(n_qubits*in_channels).to('cuda:'+str(gpu)) for gpu in range(ngpus)]
        self.convs = []
        for _ in range(self.out_channels):
            self.convs.append(New_Q_MulIn1Out_Conv1D(self.entangle_matrix, in_channels, kernel_size, stride=stride))
        self.convs = nn.ModuleList(self.convs)
        #self.weight = nn.Parameter([conv.weight for conv in self.convs])
    def forward(self, x):
        device = x.device
        output = torch.tensor([]).to(device)
        for channel in range(self.out_channels):
            conv = self.convs[channel]
            out = conv(x)
            output = torch.cat((output, out), dim=1)
            #print('out_channel: ', channel)
        output = output.squeeze(-2)
        #print('Convolution output shape', output.shape)
        return output

test = False
# Testing code
if test:
    kernel_size = 4
    n_qubits = kernel_size 
    qconv1d = QConv1D(in_channels=5, out_channels=3, kernel_size=kernel_size)
    qconv1d = qconv1d.to('cpu')
    x = torch.randn(60,5,64).to('cpu')
    start = time.time()
    out = qconv1d(x)
    stop = time.time()
    print(out.shape, '. Time taken ', stop - start, ' seconds for custom.')
    qconv1d = qconv1d.to('cuda:0')
    x = torch.randn(60,5,64).to('cuda:0')
    start = time.time()
    out = qconv1d(x)
    stop = time.time()
    print(out.shape, '. Time taken ', stop - start, ' seconds for custom.')