import torch
import numpy as np
from .Controlled_QConv_Kernel import Encoder, ConvKernel
from .QConv1D import ControlledQConv1D as Conv

in_channels = 8
out_channels = 2
kernel_size = 8

conv = Conv(in_channels, out_channels, kernel_size)

inputs = torch.randn(100, 8, 64, requires_grad=True)
output = conv(inputs)
print(output.shape)
loss = output.abs().sum()
loss.backward()
print(inputs.grad)

'''
e = Encoder(16,4,channel_ancillas=2,kernel_ancillas=1)
mag, vector = e(torch.arange(64,dtype=torch.float32).unsqueeze(0))
print(vector)
print(vector.square().sum(1))

a = torch.tensor([[1,2,3,4,5,6,7,8],[9,10,11,12,13,14,15,16],[17,18,19,20,21,22,23,24]])
a = torch.div(a, a.square().sum(1).sqrt().reshape(-1,1))
n = a.size(0)
for kernel_qubits in range(3):
    for ancillas in range(3):
        out = a.reshape(-1,2**kernel_qubits).unsqueeze(1).repeat_interleave(2**ancillas,dim=1).reshape(n, -1)/np.sqrt(2**ancillas)
        print(out, out.square().sum(1))

states = torch.randn(5,8, dtype=torch.cfloat, requires_grad=True)
angles = torch.randn(5, requires_grad=True)
out = single_qubit_z_rot(states, angles, 0)
loss = out.abs().sum()
loss.backward()
print(states.grad, angles.grad)

states = torch.randn(5,8, dtype=torch.cfloat, requires_grad=True)
inputs = torch.randn(5,3, requires_grad=True)
out = higher_order_encoding(states, inputs)
print(out.shape)
loss = out.abs().sum()
loss.backward()
print(states.grad, inputs.grad)

import torch
#from torch.utils.tensorboard import SummaryWriter
from custom_quantum_models import InputBlock
from torch_QConv_Kernel_copy import *
import numpy as np

weights = torch.cat([torch.tensor([[[1,0],[0,1]]]),torch.tensor([[[0,1],[1,0]]]),torch.tensor([[[1,2],[3,4]]])], 0)
print(weights)
print(mat_tensor_product(weights))
weights = torch.randn(5, requires_grad=True)
output = rot_mat(weights)
output = output.abs().sum()
criterion = torch.nn.MSELoss()
loss = criterion(output, torch.tensor(0,dtype=torch.float32))
loss.backward()
print(weights.grad)
ry_angles = torch.randn(1, requires_grad=True)
out = ry_mat(ry_angles)
out = out.abs().sum()
out.backward()
print(out, ry_angles.grad)


a = torch.tensor([[[1,2],[3,4]]])
b = torch.tensor([[[5,6],[7,8]]])
print(torch.matmul(a,b))

n_scalars, n_profiles,profile_size, layer_sizes, kernel_size, linear_size = 10, 2, 10, [2,3], 3, 10
ib = InputBlock(n_scalars, n_profiles,profile_size, layer_sizes, kernel_size, linear_size)
inputs = torch.randn(2, 30, requires_grad=True)
targets = torch.randn(2, 13)
criterion = torch.nn.MSELoss()
out = ib(inputs)
print(out.shape, targets.shape)
loss = criterion(out, targets)
loss.backward()
print(inputs.grad)'''