import torch
#from torch.utils.tensorboard import SummaryWriter
from custom_quantum_models import InputBlock

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