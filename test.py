import torch
#from torch.utils.tensorboard import SummaryWriter
from custom_quantum_models import InputBlock
from torch_QConv_Kernel_copy import *
import numpy as np
'''
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
'''

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
print(inputs.grad)

'''
# Writer will output to ./runs/ directory by default
writer = SummaryWriter()

model = ConvKernel(5)

inputs =  torch.randn(2, 5)
writer.add_graph(model, inputs)
writer.add_scalar('val', 1)
writer.close()


import torch
from torch import nn

class MyLinear(nn.Module):
  def __init__(self, in_features, out_features):
    super().__init__()
    self.weight = nn.Parameter(torch.randn(in_features, out_features))
    self.bias = nn.Parameter(torch.randn(out_features))

  def forward(self, input):
    return (input @ self.weight) + self.bias

k = MyLinear(10, 10)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(k.parameters(), lr=0.1, momentum=0.9)

for e in range(10000):
  input = torch.randn(2,10)
  target = torch.zeros(2,10)
  output = k(input)
  loss = criterion(output, target)
  if e%100==0:
    print(loss)
  k.zero_grad()
  loss.backward()
  optimizer.step()
  '''