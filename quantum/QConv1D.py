import torch
import torch.nn as nn

import time

from .QConv_Kernel import ConvKernel, entangle_mat
from .Easy_QConv_Kernel import ConvKernel as EasyConvKernel, entangle_mat as easy_entangle_mat
from .Dense_QConv_Kernel import ConvKernel as DenseConvKernel, MoreParamConvKernel as MoreParamDenseConvKernel, entangle_mat as dense_entangle_mat, isPowerOfTwo

# Need decompose a multichannel quantum convolution layer into many single channel quantum convolution. This is because each torch.nn.Module converted from a qml.qnode must be associated with a device that contains all the needed qubits. Having all the qubits in one layer is too many for one device.

# This kernel is for classical CNN and can be used inplace of the quantum kernel

# Multichannel input one channel output quantum convolution 2D
class Easy_Q_MulIn1Out_Conv1D(nn.Module):
    def __init__(self, entangle_matrix, in_channels, kernel_size, stride=1, initial=False):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.q_kernel = EasyConvKernel(entangle_matrix, kernel_size*in_channels, initial=initial)
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
class EasyQConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, initial=False):
        super().__init__()
        print('channels: ', in_channels, out_channels)
        self.out_channels = out_channels
        n_qubits = kernel_size
        ngpus = torch.cuda.device_count()
        self.entangle_matrix = easy_entangle_mat(n_qubits*in_channels) if ngpus == 0 else [easy_entangle_mat(n_qubits*in_channels).to('cuda:'+str(gpu)) for gpu in range(ngpus)]
        self.convs = []
        for _ in range(self.out_channels):
            self.convs.append(Easy_Q_MulIn1Out_Conv1D(self.entangle_matrix, in_channels, kernel_size, stride=stride, initial=initial))
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
class Dense_Q_MulIn1Out_Conv1D(nn.Module):
    def __init__(self, entangle_matrix, in_channels, kernel_size, ancillas=2, dilation=1, padding=0, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        if not isPowerOfTwo(kernel_size*in_channels):
            print('You need to have an input size (kernel_size*in_channels) of a power of two for efficient encoding.')
            quit()
        n_qubits = int(torch.log2(torch.tensor(kernel_size*in_channels)))+ancillas
        self.q_kernel = DenseConvKernel(entangle_matrix, n_qubits, ancillas=ancillas)
        self.unfold = nn.Unfold(kernel_size=(kernel_size,1), dilation=(dilation,1), padding=(padding,0), stride=(stride,1))
        #print('unfold ', kernel_size, dilation, padding, stride)
        #self.weight = self.q_kernel.weight
    def memory_strided_im2col(self, x):
        # x has dimension (n_batch, n_channels, length)
        x=x.unsqueeze(-1)
        out = self.unfold(x)
        out = torch.transpose(out, 1, 2)
        return out
    def forward(self, x):
        n_batch = x.size(0)
        x = self.memory_strided_im2col(x)
        #print('memory strided im2col succeed')
        x = x.reshape(-1, self.kernel_size*self.in_channels)
        output = self.q_kernel(x)
        #print('q_kernel successful with output shape ', output.shape)
        output = output.reshape(n_batch, 1, 1, -1)
        return output

# Quantum convolution 2D layer
class DenseQConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, ancillas=2, dilation=1, padding=0, stride=1):
        super().__init__()
        #print('channels: ', in_channels, out_channels)
        self.out_channels = out_channels
        n_qubits = int(torch.log2(torch.tensor(kernel_size*in_channels)))+ancillas
        ngpus = torch.cuda.device_count()
        self.entangle_matrix = dense_entangle_mat(n_qubits) if ngpus == 0 else [dense_entangle_mat(n_qubits).to('cuda:'+str(gpu)) for gpu in range(ngpus)]
        self.convs = []
        for _ in range(self.out_channels):
            self.convs.append(Dense_Q_MulIn1Out_Conv1D(self.entangle_matrix, in_channels, kernel_size, ancillas=ancillas, dilation=dilation, padding=padding, stride=stride))
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
class More_Param_Dense_Q_MulIn1Out_Conv1D(nn.Module):
    def __init__(self, entangle_matrix, in_channels, kernel_size, dilation=1, padding=0, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        if not isPowerOfTwo(kernel_size*in_channels):
            print('You need to have an input size (kernel_size*in_channels) of a power of two for efficient encoding.')
            quit()
        n_qubits = int(torch.log2(torch.tensor(kernel_size*in_channels)))+2
        self.q_kernel = MoreParamDenseConvKernel(entangle_matrix, n_qubits)
        self.unfold = nn.Unfold(kernel_size=(kernel_size,1), dilation=(dilation,1), padding=(padding,0), stride=(stride,1))
        #print('unfold ', kernel_size, dilation, padding, stride)
        #self.weight = self.q_kernel.weight
    def memory_strided_im2col(self, x):
        # x has dimension (n_batch, n_channels, length)
        x=x.unsqueeze(-1)
        out = self.unfold(x)
        out = torch.transpose(out, 1, 2)
        return out
    def forward(self, x):
        n_batch = x.size(0)
        x = self.memory_strided_im2col(x)
        #print('memory strided im2col succeed')
        x = x.reshape(-1, self.kernel_size*self.in_channels)
        output = self.q_kernel(x)
        #print('q_kernel successful with output shape ', output.shape)
        output = output.reshape(n_batch, 1, 1, -1)
        return output

# Quantum convolution 2D layer
class MoreParamDenseQConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, padding=0, stride=1):
        super().__init__()
        #print('channels: ', in_channels, out_channels)
        self.out_channels = out_channels
        n_qubits = int(torch.log2(torch.tensor(kernel_size*in_channels)))+2
        ngpus = torch.cuda.device_count()
        self.entangle_matrix = dense_entangle_mat(n_qubits) if ngpus == 0 else [dense_entangle_mat(n_qubits).to('cuda:'+str(gpu)) for gpu in range(ngpus)]
        self.convs = []
        for _ in range(self.out_channels):
            self.convs.append(More_Param_Dense_Q_MulIn1Out_Conv1D(self.entangle_matrix, in_channels, kernel_size, dilation=dilation, padding=padding, stride=stride))
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
class Q_MulIn1Out_Conv1D(nn.Module):
    def __init__(self, entangle_matrix, in_channels, kernel_size, dilation=1, padding=0, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.q_kernel = ConvKernel(entangle_matrix, kernel_size*in_channels)
        self.unfold = nn.Unfold(kernel_size=(kernel_size,1), dilation=(dilation,1), padding=(padding,0), stride=(stride,1))
        #print('unfold ', kernel_size, dilation, padding, stride)
        #self.weight = self.q_kernel.weight
    def memory_strided_im2col(self, x):
        # x has dimension (n_batch, n_channels, length)
        x=x.unsqueeze(-1)
        out = self.unfold(x)
        out = torch.transpose(out, 1, 2)
        return out
    def forward(self, x):
        n_batch = x.size(0)
        x = self.memory_strided_im2col(x)
        #print('memory strided im2col succeed')
        x = x.reshape(-1, self.kernel_size*self.in_channels)
        output = self.q_kernel(x)
        #print('q_kernel successful with output shape ', output.shape)
        output = output.reshape(n_batch, 1, 1, -1)
        return output

# Quantum convolution 2D layer
class QConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, padding=0, stride=1):
        super().__init__()
        print('channels: ', in_channels, out_channels)
        self.out_channels = out_channels
        n_qubits = kernel_size
        ngpus = torch.cuda.device_count()
        self.entangle_matrix = entangle_mat(n_qubits*in_channels) if ngpus == 0 else [entangle_mat(n_qubits*in_channels).to('cuda:'+str(gpu)) for gpu in range(ngpus)]
        self.convs = []
        for _ in range(self.out_channels):
            self.convs.append(Q_MulIn1Out_Conv1D(self.entangle_matrix, in_channels, kernel_size, dilation=dilation, padding=padding, stride=stride))
        self.convs = nn.ModuleList(self.convs)
        #self.weight = nn.Parameter([conv.weight for conv in self.convs])
    def forward(self, x):
        device = x.device
        futures = [torch.jit.fork(conv, x) for conv in self.convs]
        results = [torch.jit.wait(fut) for fut in futures]
        return torch.cat(results,1).squeeze(-2)
        #output = torch.tensor([]).to(device)
        #for channel in range(self.out_channels):
        #    conv = self.convs[channel]
        #    out = conv(x)
        #    output = torch.cat((output, out), dim=1)
        #    #print('out_channel: ', channel)
        #output = output.squeeze(-2)
        #print('Convolution output shape', output.shape)
        #return output

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