import torch
import numpy as np
from .QConv_Kernel_copy import single_qubit_z_rot, higher_order_encoding, Encoder
from .QConv1D import newQConv1D

conv = newQConv1D(1,1,2)
out = conv(torch.randn(5,1,7))
print(out)

'''
e = Encoder()
print(torch.square(e(torch.randn(2,5,dtype=torch.float32)).abs()).sum())

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
print(states.grad, inputs.grad)'''

'''import torch
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