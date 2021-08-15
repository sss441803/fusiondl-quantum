import torch
from torch._C import dtype
from torch.functional import tensordot
import torch.nn as nn
from torch.autograd import Function

import time

if torch.cuda.is_available():
    dev = "cuda:0"
else:  
    dev = "cpu"
#dev='cpu'
print(dev)
device = torch.device(dev) 

def dec2bin(n_bits, x):
    mask = 2 ** torch.arange(n_bits - 1, -1, -1).to(device)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0)

def bin2dec(n_bits, b):
    mask = 2 ** torch.arange(n_bits - 1, -1, -1).to(device)
    return torch.sum(mask * b, -1)
'''
# First bit is the most significant bit
def decimalToBinary(n_qubits: int, n: int) -> torch.Tensor:
    if n >= 2**(n_qubits):
        raise ValueError('The number exceeds the representation of the number of qubits')
    answer = torch.zeros(n_qubits, dtype=torch.bool)
    rem = n
    for qubit in range(n_qubits):
        if rem >= 2**(n_qubits-qubit-1):
            answer[qubit] = True
            rem = rem - 2**(n_qubits-qubit-1)
    return answer

# First bit is the most significant bit
def binaryToDecimal(binary_tensor: torch.Tensor) -> int:
    n_qubits = binary_tensor.size(0)
    answer = 0
    for qubit in range(n_qubits):
        answer += 2**(n_qubits-qubit-1) if binary_tensor[qubit] == True else 0
    return answer
'''

'''Creates the encoded n qubit state vector based on variation encoding'''
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, ry_angles: torch.Tensor, rz_angles: torch.Tensor) -> torch.Tensor:
        ry_angles = ry_angles/2
        rz_angles = rz_angles/2
        '''Parallel compute entries for the ry matrix'''
        ry_entry11 = torch.cos(ry_angles).unsqueeze(-1).to(device)
        ry_entry12 = -torch.sin(ry_angles).unsqueeze(-1).to(device)
        ry_entry21 = torch.sin(ry_angles).unsqueeze(-1).to(device)
        ry_entry22 = torch.cos(ry_angles).unsqueeze(-1).to(device)

        '''Parallel compute entries for the rz matrix'''
        rz_entry11 = torch.exp(-1j*rz_angles).unsqueeze(-1).to(device)
        #rz_entry12 = torch.zeros(n_batch, n_qubits).unsqueeze(-1)
        #rz_entry21 = torch.zeros(n_batch, n_qubits).unsqueeze(-1)
        rz_entry22 = torch.exp(1j*rz_angles).unsqueeze(-1).to(device)
        '''Parallel compute entries for the composite rz*ry matrix'''
        entry11 = ry_entry11 * rz_entry11
        entry12 = ry_entry12 * rz_entry11
        entry21 = ry_entry21 * rz_entry22
        entry22 = ry_entry22 * rz_entry22

        mat = torch.cat((entry11, entry12, entry21, entry22), -1) # mat.shape (n_batch, n_qubits, 4)

        n_batch, n_qubits = torch.tensor(mat.size(0)).to(device), torch.tensor(mat.size(1)).to(device)
        #tensor_mat = torch.zeros(n_batch, 2**n_qubits, 2**n_qubits, dtype=torch.cfloat)
        '''Initialize the state vector for encoding'''
        vector = torch.zeros(n_batch, 2**n_qubits, dtype=torch.cfloat).to(device)
        '''Since the initial state is |00...000>, i.e. (1,0,0,...), the first column of the rz*ry matrix is the result'''
        for row in torch.arange(2**n_qubits).to(device):
            row_bin = dec2bin(n_qubits, row) # Returns a tensor of bool that stands for the basis for the nth qubit
            #print('row_bin ', row_bin)
            #for col in range(2**n_qubits):
            #    col_bin = bin(col)[2:]
            #    col_bin = '0'*(n_qubits-len(col_bin))+col_bin
                #print('col_bin ', col_bin)
            #    entries = torch.cat([mat[:, qubit, 2*int(row_bin[qubit])+int(col_bin[qubit])].unsqueeze(1) for qubit in range(n_qubits)], 1)
            #    tensor_mat[:, row, col] = torch.prod(entries, dim=1)

            col = torch.tensor(0).to(device)
            col_bin = dec2bin(n_qubits, col)
            #print('col_bin ', col_bin)
            val_list = [mat[:, qubit, 2*row_bin[qubit]+col_bin[qubit]].unsqueeze(1) for qubit in range(n_qubits)] # List of values that needs to be multiplied to obtain the row_th row of the resulting state vector. 2*int(row_bin[qubit])+int(col_bin[qubit]) computes which entry of the 2 by 2 matrix for this qubit should be chosen.
            entries = torch.cat(val_list, 1)
            vector[:, row] = torch.prod(entries, dim=1)
        return vector

def col_vector_constructor(n_qubits: int, col: int, c_bit: int, t_bit: int) -> torch.Tensor:
    # Initialize output col vector
    vector = torch.zeros(2**n_qubits, dtype=torch.bool).to(device)
    col_bin = dec2bin(n_qubits, col)
    # Implement CNOT
    if col_bin[c_bit]:
        col_bin[t_bit] = not col_bin[t_bit] # Flips 0 and 1
    # Convert qubit_states into tensor basis state
    vector[bin2dec(n_qubits, col_bin)] = 1
    return vector

def CNOT_mat(n_qubits: int, c_bit: int, t_bit: int) -> torch.Tensor:
        cnot_mat =  torch.zeros(2**n_qubits, 2**n_qubits).to(device)
        for col in torch.arange(2**n_qubits).to(device):
            cnot_mat[:, col] = col_vector_constructor(n_qubits, col, c_bit, t_bit)
        return cnot_mat

def entangle_mat(n_qubits: int) -> torch.Tensor:
    for qubit in torch.arange(n_qubits):
        mat = torch.matmul(CNOT_mat(n_qubits, qubit, (qubit+1)%n_qubits), mat) if qubit != 0 else CNOT_mat(n_qubits, 0, 1)
    return mat.cfloat()

def list_mat_mul(mat_list1: list, mat_list2: list) -> list:
    final_list = [[11,12],[21,22]]
    final_list[0][0] = mat_list1[0][0]*mat_list2[0][0] + mat_list1[0][1]*mat_list2[1][0]
    final_list[0][1] = mat_list1[0][0]*mat_list2[0][1] + mat_list1[0][1]*mat_list2[1][1]
    final_list[1][0] = mat_list1[1][0]*mat_list2[0][0] + mat_list1[1][1]*mat_list2[1][0]
    final_list[1][1] = mat_list1[1][0]*mat_list2[0][1] + mat_list1[1][1]*mat_list2[1][1]
    return final_list

def rx_mat_list(theta: nn.Parameter) -> list:
    angle = theta/2
    mat_list = [[torch.cos(angle).to(device), -1j*torch.sin(angle).to(device)], [-1j*torch.sin(angle).to(device), torch.cos(angle).to(device)]]
    return mat_list

def ry_mat_list(theta: nn.Parameter) -> list:
    angle = theta/2
    mat_list = [[torch.cos(angle).to(device), -torch.sin(angle).to(device)], [torch.sin(angle).to(device), torch.cos(angle).to(device)]]
    return mat_list

def rz_mat_list(theta: nn.Parameter) -> list:
    angle = theta/2
    mat_list = [[torch.exp(-1j*angle).to(device), torch.tensor(0.0).to(device)], [torch.tensor(0.0).to(device), torch.exp(1j*angle).to(device)]]
    return mat_list

def rot_mat(n_qubits: int, thetas: nn.Parameter) -> torch.Tensor:
    rx_matrix_list = ry_mat_list(thetas[0])
    ry_matrix_list = ry_mat_list(thetas[1])
    rz_matrix_list = ry_mat_list(thetas[2])
    rot_matrix_list = list_mat_mul(ry_matrix_list, rx_matrix_list)
    rot_matrix_list = list_mat_mul(rz_matrix_list, rot_matrix_list)
    identity_mat = torch.eye(2**(n_qubits-1), 2**(n_qubits-1), dtype = torch.cfloat, requires_grad=False).to(device)
    tensor_mat_top = torch.cat((rot_matrix_list[0][0]*identity_mat, rot_matrix_list[0][1]*identity_mat), 1)
    tensor_mat_bottom  = torch.cat((rot_matrix_list[1][0]*identity_mat, rot_matrix_list[1][1]*identity_mat), 1)
    tensor_mat = torch.cat((tensor_mat_top, tensor_mat_bottom), 0)
    return tensor_mat

class ConvKernel(nn.Module):
    def __init__(self, n_qubits):
        super().__init__()
        self.n_qubits = torch.tensor(n_qubits).to(device)
        self.encoding = Encoder()
        self.entangle_matrix = entangle_mat(self.n_qubits) # entangle_matrix.shape (2**n_qubits, 2**n_qubits)
        self.weights = nn.Parameter(torch.randn(3, dtype=torch.cfloat))#.to(device)
    def forward(self, inputs):
        # inputs.shape (n_batch, n_qubits)
        '''Calculate angles of rotation for variational encoding'''
        ry_angles = torch.arctan(inputs)
        rz_angles = torch.arctan(torch.square(inputs))
        '''Calculate rotated states'''
        vector = self.encoding(ry_angles, rz_angles) # vector.shape (n_batch, 2**n_qubits)
        '''Entangling'''
        vector = torch.tensordot(vector, self.entangle_matrix, dims=([-1],[-1]))
        '''Final rotation'''
        rot_matrix = rot_mat(self.n_qubits, self.weights)
        vector = torch.tensordot(vector, rot_matrix, dims=([-1],[-1]))
        del rot_matrix
        '''Measurement'''
        # The expectation value of <vector|Z0|vector> is 1*sum_of_mag_for_the_first_half_of_amplitudes + 0*sum_of_mag_for_the_second_half_of_amplitudes
        vector[:, 2**(self.n_qubits-1):] = 0
        return torch.sum(torch.square(vector.abs()), -1)
        #return self.weights[0]

'''

n_batch = 100000
n_qubits = 5


device = torch.device('cpu') 
k=ConvKernel(5)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(k.parameters(), lr=0.001, momentum=0.9)

input = torch.ones(n_batch, n_qubits).to(device)
target = torch.zeros(n_batch).to(device)

start = time.time()

for e in range(100):
    output = k(input)
    loss = criterion(output, target)
    if e%100==0:
        print(loss)
        k.zero_grad()
    loss.backward()
    optimizer.step()

stop = time.time()
print(stop - start, ' seconds for cpu')

device = torch.device('cuda:0') 
k=ConvKernel(5)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(k.parameters(), lr=0.001, momentum=0.9)

input = torch.ones(n_batch, n_qubits).to(device)
target = torch.zeros(n_batch).to(device)

start = time.time()

for e in range(100):
    output = k(input)
    loss = criterion(output, target)
    if e%100==0:
        print(loss)
        k.zero_grad()
    loss.backward()
    optimizer.step()

stop = time.time()
print(stop - start, ' seconds for gpu')
'''

'''def rx_mat(theta: nn.Parameter) -> torch.Tensor:
    angle = theta/2
    mat = torch.tensor([[torch.cos(angle), -1j*torch.sin(angle)],
                        [-1j*torch.sin(angle), torch.cos(angle)]], dtype = torch.cfloat, requires_grad=True)
    return mat

def ry_mat(theta: nn.Parameter) -> torch.Tensor:
    angle = theta/2
    mat = torch.tensor([[torch.cos(angle), -torch.sin(angle)],
                        [torch.sin(angle), torch.cos(angle)]], dtype = torch.cfloat, requires_grad=True)
    return mat

def rz_mat(theta: nn.Parameter) -> torch.Tensor:
    angle = theta/2
    mat = torch.tensor([[torch.exp(-1j*angle), 0],
                        [0, torch.exp(1j*angle)]], dtype = torch.cfloat, requires_grad=True)
    return mat

def rot_mat(n_qubits: int, thetas: nn.Parameter) -> torch.Tensor:
    single_mat = torch.matmul(ry_mat(thetas[1]), rx_mat(thetas[0]))
    single_mat = torch.matmul(rz_mat(thetas[2]), single_mat)
    #single_mat = torch.tensor([[1,2],[3,4]], dtype=torch.cfloat)
    identity_mat = torch.eye(2**(n_qubits-1), 2**(n_qubits-1), dtype = torch.cfloat, requires_grad=False)
    tensor_mat_top = torch.cat((single_mat[0,0]*identity_mat, single_mat[0,1]*identity_mat), 1)
    tensor_mat_bottom  = torch.cat((single_mat[1,0]*identity_mat, single_mat[1,1]*identity_mat), 1)
    tensor_mat = torch.cat((tensor_mat_top, tensor_mat_bottom), 0)
    return tensor_mat'''

'''
def forward(inputs):
    n_qubits = inputs.size(0)
    # Calculate angles of rotation for variational encoding
    ry_angles = torch.arctan(inputs)
    rz_angles = torch.arctan(inputs*inputs)
    # Initialize the collection of individual rotated states
    zero_state = torch.tensor([[1],[0]], dtype=torch.cfloat)
    states = []
    # Calculate rotated states
    for qubit in range(n_qubits):
        state = torch.matmul(ry_mat(ry_angles[qubit]), zero_state)
        state = torch.matmul(rz_mat(rz_angles[qubit]), state)
        states.append(state)
    # Combine individual states into a tensor state
    for qubit in range(n_qubits):
        if qubit == 0:
            tensor_state_after_encoding = states[qubit]
        else:
            tensor_state_after_encoding = torch.reshape(torch.tensordot(tensor_state_after_encoding, states[qubit], dims=([1],[1])), (-1, 1))
    return tensor_state_after_encoding

# Inherit from Function
class LinearFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias

def mat_tensor_product(mats: list) -> torch.Tensor:
    n_qubits = len(mats)
    for qubit in range(n_qubits):
        mat = torch.unsqueeze(mats[qubit], -1)
        if qubit == 0:
            tensor_mat = mat
        else:
            tensor_mat = torch.tensordot(tensor_mat, mat, dims=([-1],[-1]))
            tensor_mat = torch.reshape(tensor_mat, (tensor_mat.size(0)*mat.size(0), -1, 1))
    return tensor_mat.squeeze(-1)

def vector_tensor_product(states: list) -> torch.Tensor:
    n_qubits = len(states)
    for qubit in range(n_qubits):
        state = torch.unsqueeze(states[qubit], -1)
        if qubit == 0:
            tensor_state = state
        else:
            tensor_state = torch.reshape(torch.tensordot(tensor_state, state, dims=([-1],[-1])), (-1, 1))
            tensor_state = torch.unsqueeze(tensor_state, -1)
    return tensor_state.squeeze(-1)

def separate_to_tensor_mat(mat):
    n_batch, n_qubits = mat.size(0), mat.size(1)
    tensor_mat = torch.zeros(n_batch, 2**n_qubits, 2**n_qubits, dtype=torch.cfloat)
    for row in range(2**n_qubits):
        row_bin = bin(row)[2:]
        row_bin = '0'*(n_qubits-len(row_bin))+row_bin
        #print('row_bin ', row_bin)
        for col in range(2**n_qubits):
            col_bin = bin(col)[2:]
            col_bin = '0'*(n_qubits-len(col_bin))+col_bin
            #print('col_bin ', col_bin)
            entries = torch.cat([mat[:, qubit, 2*int(row_bin[qubit])+int(col_bin[qubit])].unsqueeze(1) for qubit in range(n_qubits)], 1)
            tensor_mat[:, row, col] = torch.prod(entries, dim=1)
    return tensor_mat

def rx_mat(theta):
    angle = theta/2
    entry11 = torch.cos(angle).unsqueeze(-1)
    entry12 = -1j*torch.sin(angle).unsqueeze(-1)
    entry21 = -1j*torch.sin(angle).unsqueeze(-1)
    entry22 = torch.cos(angle).unsqueeze(-1)
#    entry11 = torch.ones(n_batch, n_qubits).unsqueeze(-1)
#    entry12 = 2*torch.ones(n_batch, n_qubits).unsqueeze(-1)
#    entry21 = 3*torch.ones(n_batch, n_qubits).unsqueeze(-1)
#    entry22 = 4*torch.ones(n_batch, n_qubits).unsqueeze(-1)
    mat = torch.cat((entry11, entry12, entry21, entry22), -1) # mat.shape (n_batch, n_qubits, 4)
    tensor_mat = separate_to_tensor_mat(mat)
    return tensor_mat

def ry_mat(theta):
    angle = theta/2
    entry11 = torch.cos(angle).unsqueeze(-1)
    entry12 = -torch.sin(angle).unsqueeze(-1)
    entry21 = torch.sin(angle).unsqueeze(-1)
    entry22 = torch.cos(angle).unsqueeze(-1)
    mat = torch.cat((entry11, entry12, entry21, entry22), -1)
    tensor_mat = separate_to_tensor_mat(mat)
    return tensor_mat

def rz_mat(theta):
    n_batch = theta.size(0)
    n_qubits = theta.size(1)
    angle = theta/2
    entry11 = torch.exp(-1j*angle).unsqueeze(-1)
    entry12 = torch.zeros(n_batch, n_qubits).unsqueeze(-1)
    entry21 = torch.zeros(n_batch, n_qubits).unsqueeze(-1)
    entry22 = torch.exp(1j*angle).unsqueeze(-1)
    mat = torch.cat((entry11, entry12, entry21, entry22), -1)
    tensor_mat = separate_to_tensor_mat(mat)
    return tensor_mat
'''