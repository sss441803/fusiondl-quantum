import torch
import torch.nn as nn
import pennylane as qml

# Need decompose a multichannel quantum convolution layer into many single channel quantum convolution. This is because each torch.nn.Module converted from a qml.qnode must be associated with a device that contains all the needed qubits. Having all the qubits in one layer is too many for one device.

# This kernel is for classical CNN and can be used inplace of the quantum kernel
class kernel_module(nn.Module):
    def __init__(self, kernel_size):
        super(kernel_module, self).__init__()
        self.kernel_size = kernel_size
        self.kernel = torch.randn(kernel_size)
        self.flat_kernel = self.kernel.reshape(-1)
    def forward(self, x):
        out = torch.tensordot(x, self.flat_kernel, dims=([-1],[0]))
        return out

# defines the quantum convolutional kernel (a variational circuit)
def build_qconv(dev, kernel_size) -> nn.Module:
    n_qubits = kernel_size[0]*kernel_size[1]
    # Defines the q_node that is the quantum convolution kernel
    @qml.qnode(dev, interface='torch', diff_method='adjoint')
    def circuit(inputs, weights):
            qml.templates.AngleEmbedding(torch.arctan(inputs), rotation='Y', wires=range(n_qubits))
            qml.templates.AngleEmbedding(torch.arctan(inputs*inputs), rotation='Z', wires=range(n_qubits))
            if n_qubits != 1:
                if n_qubits == 2:
                    qml.CNOT(wires=[0, 1])
                else:
                    for w in range(n_qubits):
                        qml.CNOT(wires=[w,(w+1)%n_qubits])
            qml.Rot(*weights, wires=0)
            return qml.expval(qml.PauliZ(wires=0))
    weight_shapes = {"weights": 3} # weights are the rotation angles for each qubit. There are 3 angles to fully specify the rotation
    # Make the kernel a torch.nn.Module object
    q_kernel = qml.qnn.TorchLayer(circuit, weight_shapes)
    return q_kernel

# Initialize convolutional layer with the custom kernel
class CustomKernel_1In1Out_Conv2D(nn.Module):
    def __init__(self, kernel_module: nn.Module, kernel_size, stride=(1,1)): #(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
        super(CustomKernel_1In1Out_Conv2D, self).__init__()
        self.kernel_module = kernel_module
        self.kernel_size = kernel_size
        self.stride = stride
    def memory_strided_im2col(self, x, kernel_size, stride):
        # x has dimension (n_batch, n_channels, im_width, im_height)
        output_shape = [0, 0]
        output_shape[0] = x.shape[-2] - kernel_size[0] + 1
        output_shape[1] = x.shape[-1] - kernel_size[1] + 1
        out = x.unfold(-2,kernel_size[0],stride[0]).unfold(-2,kernel_size[1],stride[1])
        #print('shape after unfold: ', out.shape)
        out = out.reshape(x.shape[0], output_shape[0]*output_shape[1], kernel_size[0]*kernel_size[1])
        #print('shape after reshape: ', out.shape)
        return out
    def forward(self, x):
        output_shape = [0, 0]
        output_shape[0] = x.shape[-2] - self.kernel_size[0] + 1
        output_shape[1] = x.shape[-1] - self.kernel_size[1] + 1
        x = self.memory_strided_im2col(x, self.kernel_size, self.stride)
        out = self.kernel_module(x)
        out = out.reshape(x.size(0), *output_shape)
        return out

# 1In1Out channel quantum convolution 2D
class Q_1In1Out_Conv2D(nn.Module):
    def __init__(self, dev, kernel_size, stride=(1,1)):
        super(Q_1In1Out_Conv2D, self).__init__()
        q_kernel = build_qconv(dev, kernel_size)
        #q_kernel = kernel_module(kernel_size)
        self.conv = CustomKernel_1In1Out_Conv2D(q_kernel, kernel_size, stride=stride)
    def forward(self, x):
        out = self.conv(x)
        return out

# Multichannel input one channel output quantum convolution 2D
class Q_MulIn1Out_Conv2D(nn.Module):
    def __init__(self, dev, in_channels, kernel_size, stride=(1,1)):
        super(Q_MulIn1Out_Conv2D, self).__init__()
        self.in_channels = in_channels
        self.linear = nn.Linear(in_channels, 1)
        self.convs = []
        for _ in range(self.in_channels):
            self.convs.append(Q_1In1Out_Conv2D(dev, kernel_size, stride))
    def forward(self, x):
        output = torch.tensor([])
        for channel in range(self.in_channels):
            x_channel = x[:, channel]
            conv = self.convs[channel]
            out = conv(x_channel)
            out = out.unsqueeze(1)
            output = torch.cat((output, out), dim=1)
        output = self.linear(torch.transpose(output, 1, -1))
        output = torch.transpose(output, 1, -1)
        return output

# Quantum convolution 2D layer
class QConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1)):
        super(QConv2D, self).__init__()
        self.out_channels = out_channels
        n_qubits = kernel_size[0]*kernel_size[1]
        dev = qml.device("default.qubit", wires=n_qubits)
        self.convs = []
        for _ in range(self.out_channels):
            self.convs.append(Q_MulIn1Out_Conv2D(dev, in_channels, kernel_size, stride))
    def forward(self, x):
        output = torch.tensor([])
        for channel in range(self.out_channels):
            conv = self.convs[channel]
            out = conv(x)
            output = torch.cat((output, out), dim=1)
        return output

# Testing code
#kernel_size = (3,3)
#n_qubits = kernel_size[0]*kernel_size[1]
#dev = qml.device("default.qubit", wires=n_qubits) 
#qconv2d = QConv2D(in_channels=3, out_channels=3, kernel_size=kernel_size)
#x = torch.randn(2,3,20,19)
#out = qconv2d(x)
#print(out.shape)