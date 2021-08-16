import torch
import torch.nn as nn

import time

from torch_QConv_Kernel import ConvKernel

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
        x = x.reshape(-1, self.kernel_size)
        out = self.kernel_module(x)
        out = out.reshape(n_batch, -1, output_shape)
        return out
    def to(self, *args, **kwargs):
        print('Q_MulIn1Out_Conv1D to device')
        self = super().to(*args, **kwargs)
        self.kernel_module = self.kernel_module.to(*args, **kwargs)
        return self

# 1In1Out channel quantum convolution 2D
class Q_1In1Out_Conv1D(nn.Module):
    def __init__(self, kernel_size, stride=(1,1)):
        super(Q_1In1Out_Conv1D, self).__init__()
        q_kernel = ConvKernel(kernel_size)
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
class Q_MulIn1Out_Conv1D(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1):
        super(Q_MulIn1Out_Conv1D, self).__init__()
        self.in_channels = in_channels
        self.linear = nn.Linear(in_channels, 1)
        self.convs = []
        for _ in range(self.in_channels):
            self.convs.append(Q_1In1Out_Conv1D(kernel_size, stride))
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
        return output
    def to(self, *args, **kwargs):
        print('Q_MulIn1Out_Conv1D to device')
        self = super().to(*args, **kwargs)
        self.linear = self.linear.to(*args, **kwargs)
        for channel in range(self.in_channels):
            self.convs[channel].to(*args, **kwargs)
        return self

# Quantum convolution 2D layer
class QConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(QConv1D, self).__init__()
        self.out_channels = out_channels
        n_qubits = kernel_size
        self.convs = []
        for _ in range(self.out_channels):
            self.convs.append(Q_MulIn1Out_Conv1D(in_channels, kernel_size, stride))
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
        return output
    def to(self, *args, **kwargs):
        print('QConv1D to device')
        self = super().to(*args, **kwargs) 
        for channel in range(self.out_channels):
            self.convs[channel].to(*args, **kwargs)
        return self

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