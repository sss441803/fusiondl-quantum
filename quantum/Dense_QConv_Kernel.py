import torch
import torch.nn as nn
import torch.utils.checkpoint
import numpy as np
import math

pi = np.pi

def dec2bin(n_bits, x):
    device = x.device
    mask = 2 ** torch.arange(n_bits - 1, -1, -1).to(device)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0)

def bin2dec(n_bits, b):
    device = b.device
    mask = 2 ** torch.arange(n_bits - 1, -1, -1).to(device)
    return torch.sum(mask * b, -1)

def isPowerOfTwo(x):
    return (x and (not(x & (x - 1))) )

'''Amplitude encoding'''
class Encoder(nn.Module):
    def __init__(self, ancillas=2):
        super().__init__()
        self.ancillas = ancillas
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        device = inputs.device
        num_basis = inputs.size(1)
        if not isPowerOfTwo(num_basis):
            print('You need to have an input size of a power of two for efficient encoding.')
            quit()
        top = torch.ones( (inputs.size(0), inputs.size(1)*(2**self.ancillas - 1)) ).to(device)
        vector = torch.cat((top, inputs), 1)
        vector = torch.div(vector, vector.square().sum(1).sqrt().reshape(-1,1))
        return vector.cfloat()

'''Execute single qubit rotation on a tensor product state'''
def single_qubit_z_rot(states, angles, qubit) -> torch.Tensor:
    rows = states.size(1)
    n_qubits = int(math.log2(rows))

    for row in torch.arange(rows):
        row_bin = dec2bin(n_qubits, row)
        if row_bin[qubit]:
            results = (torch.exp(-1j*angles/2)*states[:,row]).unsqueeze(-1) if row==0 else torch.cat([results, (torch.exp(-1j*angles/2)*states[:,row]).unsqueeze(-1)], -1)
        else:
            results = (torch.exp(1j*angles/2)*states[:,row]).unsqueeze(-1) if row==0 else torch.cat([results, (torch.exp(1j*angles/2)*states[:,row]).unsqueeze(-1)], -1)
    return results

'''Encodes the quantum state with hard to simulate higher order encoding'''
'''Follows arxiv:2011.00027 fig 4'''
def higher_order_encoding(states, inputs) -> torch.Tensor:
    device = inputs.device
    rows = states.size(1)
    n_qubits = int(math.log2(rows))
    for c_bit in range(n_qubits):
        for t_bit in range(c_bit+1, n_qubits, 1):
            CNOT_matrix = CNOT_mat(n_qubits, c_bit, t_bit).cfloat().to(device)
            states = torch.tensordot(states, CNOT_matrix, dims=([-1], [-1]))
            #states = torch.utils.checkpoint.checkpoint(torch.tensordot, states, CNOT_matrix, ([-1], [-1]))
            states = single_qubit_z_rot(states, inputs[:, c_bit]*inputs[:, t_bit]*2*pi, t_bit)
            states = torch.tensordot(states, CNOT_matrix, dims=([-1], [-1]))
            #states = torch.utils.checkpoint.checkpoint(torch.tensordot, states, CNOT_matrix, ([-1], [-1]))
    return states

def CNOT_col_vector_constructor(n_qubits: int, col: int, c_bit: int, t_bit: int) -> torch.Tensor:
    # Initialize output col vector
    vector = torch.zeros(2**n_qubits, dtype=torch.bool)
    col_bin = dec2bin(n_qubits, col)
    # Implement CNOT
    if col_bin[c_bit]:
        col_bin[t_bit] = not col_bin[t_bit] # Flips 0 and 1
    # Convert qubit_states into tensor basis state
    vector[bin2dec(n_qubits, col_bin)] = 1
    return vector

def CNOT_mat(n_qubits: int, c_bit: int, t_bit: int) -> torch.Tensor:
    cnot_mat =  torch.zeros(2**n_qubits, 2**n_qubits)
    for col in torch.arange(2**n_qubits):
        cnot_mat[:, col] = CNOT_col_vector_constructor(n_qubits, col, c_bit, t_bit)
    return cnot_mat

def entangle_mat(n_qubits: int) -> torch.Tensor:
    for c_bit in torch.arange(n_qubits):
        for t_bit in torch.arange(c_bit+1, n_qubits, 1):
            if c_bit == 0 and t_bit == 1:
                mat = CNOT_mat(n_qubits, c_bit, t_bit)
            else:
                mat = torch.matmul(CNOT_mat(n_qubits, c_bit, t_bit), mat)
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
    mat_list = [[torch.cos(angle), -1j*torch.sin(angle)], [-1j*torch.sin(angle), torch.cos(angle)]]
    return mat_list

def ry_mat_list(theta: nn.Parameter) -> list:
    angle = theta/2
    mat_list = [[torch.cos(angle), -torch.sin(angle)], [torch.sin(angle), torch.cos(angle)]]
    return mat_list

def rz_mat_list(theta: nn.Parameter) -> list:
    angle = theta/2
    mat_list = [[torch.exp(-1j*angle), torch.tensor(0.0)], [torch.tensor(0.0), torch.exp(1j*angle)]]
    return mat_list

def final_rot_mat(n_qubits: int, thetas: nn.Parameter) -> torch.Tensor:
    rx_matrix_list = ry_mat_list(thetas[0]*2*pi)
    ry_matrix_list = ry_mat_list(thetas[1]*2*pi)
    rz_matrix_list = ry_mat_list(thetas[2]*2*pi)
    rot_matrix_list = list_mat_mul(ry_matrix_list, rx_matrix_list)
    rot_matrix_list = list_mat_mul(rz_matrix_list, rot_matrix_list)
    identity_mat = torch.eye(2**(n_qubits-1), 2**(n_qubits-1), dtype = torch.cfloat, requires_grad=False).to(thetas.device)
    tensor_mat_top = torch.cat((rot_matrix_list[0][0]*identity_mat, rot_matrix_list[0][1]*identity_mat), 1)
    tensor_mat_bottom  = torch.cat((rot_matrix_list[1][0]*identity_mat, rot_matrix_list[1][1]*identity_mat), 1)
    tensor_mat = torch.cat((tensor_mat_top, tensor_mat_bottom), 0)
    return tensor_mat

def mat_tensor_product(mats: torch.Tensor) -> torch.Tensor:
    if mats.shape[0] == 1:
        return mats[0]
    else:
        rest_tensor_mat = mat_tensor_product(mats[1:])
        tensor_mat = torch.cat(
                                (torch.cat(
                                    (mats[0][0,0]*rest_tensor_mat, mats[0][0,1]*rest_tensor_mat),
                                1),
                                torch.cat(
                                    (mats[0][1,0]*rest_tensor_mat, mats[0][1,1]*rest_tensor_mat),
                                1)),
                            0)
        return tensor_mat

def rx_mat(theta: nn.Parameter) -> torch.Tensor:
    angle = theta/2
    entry11 = torch.cos(angle+0j).unsqueeze(0)
    entry12 = -1j*torch.sin(angle+0j).unsqueeze(0)
    entry21 = -1j*torch.sin(angle+0j).unsqueeze(0)
    entry22 = torch.cos(angle+0j).unsqueeze(0)
    mat = torch.cat(
                    (
                    torch.cat((entry11, entry12)).unsqueeze(0),
                    torch.cat((entry21, entry22)).unsqueeze(0)
                    ), 
                    0)
    return mat

def ry_mat(theta: nn.Parameter) -> torch.Tensor:
    angle = theta/2
    entry11 = torch.cos(angle+0j).unsqueeze(0)
    entry12 = -torch.sin(angle+0j).unsqueeze(0)
    entry21 = torch.sin(angle+0j).unsqueeze(0)
    entry22 = torch.cos(angle+0j).unsqueeze(0)
    mat = torch.cat(
                    (
                    torch.cat((entry11, entry12)).unsqueeze(0),
                    torch.cat((entry21, entry22)).unsqueeze(0)
                    ), 
                    0)
    return mat

def rz_mat(theta: nn.Parameter) -> torch.Tensor:
    angle = theta/2
    entry11 = torch.exp(-1j*angle).unsqueeze(0)
    entry12 = torch.tensor(0, dtype=torch.cfloat).unsqueeze(0).to(angle.device)
    entry21 = torch.tensor(0, dtype=torch.cfloat).unsqueeze(0).to(angle.device)
    entry22 = torch.exp(1j*angle).unsqueeze(0)
    mat = torch.cat(
                    (
                    torch.cat((entry11, entry12)).unsqueeze(0),
                    torch.cat((entry21, entry22)).unsqueeze(0)
                    ), 
                    0)
    return mat

def rot_mat(weights: nn.Parameter) -> torch.Tensor:
    n_qubits = weights.shape[0]
    mats = torch.cat([ry_mat(weights[qubit]*2*pi).unsqueeze(0) for qubit in range(n_qubits)], 0)
    rot_matrix = mat_tensor_product(mats)
    return rot_matrix

def more_param_rot_mat(weights: nn.Parameter) -> torch.Tensor:
    n_qubits = weights.shape[0]//2
    mats = torch.cat([torch.matmul(rx_mat(weights[qubit]*2*pi), rz_mat(weights[n_qubits+qubit]*2*pi)).unsqueeze(0) for qubit in range(n_qubits)], 0)
    rot_matrix = mat_tensor_product(mats)
    return rot_matrix

def parity_vec(n_qubits) -> torch.Tensor:
    vector = -torch.ones(2**n_qubits)
    for row in torch.arange(2**n_qubits):
        row_bin = dec2bin(n_qubits, row)
        vector[row] = vector[row] + 2*(row_bin.sum()%2) # Parity
    return vector

class ConvKernel(nn.Module):
    def __init__(self, entangle_matrix, n_qubits, ancillas=2):
        super().__init__()
        # Determine how to embed the angles
        self.n_qubits = n_qubits
        self.encoding = Encoder(ancillas=ancillas)
        self.weight = nn.Parameter(torch.randn(3+2*n_qubits, dtype=torch.float32))
        self.entangle_matrix = entangle_matrix
        self.parity_vector = parity_vec(n_qubits)
    def forward(self, inputs):
        # inputs.shape (n_batch, 2**n_qubits)
        device = str(inputs.device)
        #if inputs.max() > 1:
        #    inputs = torch.arctan(inputs)
        '''Calculate rotated states'''
        vector = self.encoding(inputs) # vector.shape (n_batch, n_qubits)
        '''Product of entangle matrix and rotation matrix'''
        rot_matrix1 = rot_mat(self.weight[3:3+self.n_qubits])
        rot_matrix2 = rot_mat(self.weight[3+self.n_qubits:])
        if device == 'cpu':
            matrix = torch.chain_matmul(self.entangle_matrix, rot_matrix1, self.entangle_matrix, rot_matrix2, self.entangle_matrix)
        else:
            matrix = torch.chain_matmul(self.entangle_matrix[int(device[-1])], rot_matrix1, self.entangle_matrix[int(device[-1])], rot_matrix2, self.entangle_matrix[int(device[-1])])
        '''Rotate then Entangle'''
        vector = torch.tensordot(vector, matrix, dims=([-1],[-1]))
        #vector = torch.utils.checkpoint.checkpoint(torch.tensordot, vector, matrix, ([-1], [-1]))
        '''Final rotation'''
        final_rot_matrix = final_rot_mat(self.n_qubits, self.weight[:3])
        vector = torch.tensordot(vector, final_rot_matrix, dims=([-1],[-1]))
        #vector = torch.utils.checkpoint.checkpoint(torch.tensordot, vector, final_rot_matrix, ([-1], [-1]))
        '''Measurement'''
        # The expectation value of <vector|Z0|vector> is 1*sum_of_mag_for_the_first_half_of_amplitudes + 0*sum_of_mag_for_the_second_half_of_amplitudes
        #vector[:, 2**(self.n_qubits-1):] = 0
        #return torch.sum(torch.square(vector.abs()), -1)
        #return torch.sum(torch.square(vector[:, 2**(self.n_qubits-1):].abs()), -1) - torch.sum(torch.square(vector[:, :2**(self.n_qubits-1)].abs()), -1)
        #print(vector.square().sum()/inputs.size(0))
        return torch.tensordot(vector.abs().square(), self.parity_vector.to(inputs.device), dims=([-1],[-1]))

class MoreParamConvKernel(nn.Module):
    def __init__(self, entangle_matrix, n_qubits):
        super().__init__()
        # Determine how to embed the angles
        self.n_qubits = n_qubits
        self.encoding = Encoder()
        self.weight = nn.Parameter(torch.randn(3+4*n_qubits, dtype=torch.float32))
        self.entangle_matrix = entangle_matrix
        self.parity_vector = parity_vec(n_qubits)
    def forward(self, inputs):
        # inputs.shape (n_batch, 2**n_qubits)
        device = str(inputs.device)
        #if inputs.max() > 1:
        #    inputs = torch.arctan(inputs)
        '''Calculate rotated states'''
        vector = self.encoding(inputs) # vector.shape (n_batch, n_qubits)
        '''Product of entangle matrix and rotation matrix'''
        rot_matrix1 = more_param_rot_mat(self.weight[3:3+2*self.n_qubits])
        rot_matrix2 = more_param_rot_mat(self.weight[3+2*self.n_qubits:])
        if device == 'cpu':
            matrix = torch.chain_matmul(self.entangle_matrix, rot_matrix1, self.entangle_matrix, rot_matrix2, self.entangle_matrix)
        else:
            matrix = torch.chain_matmul(self.entangle_matrix[int(device[-1])], rot_matrix1, self.entangle_matrix[int(device[-1])], rot_matrix2, self.entangle_matrix[int(device[-1])])
        '''Rotate then Entangle'''
        vector = torch.tensordot(vector, matrix, dims=([-1],[-1]))
        #vector = torch.utils.checkpoint.checkpoint(torch.tensordot, vector, matrix, ([-1], [-1]))
        '''Final rotation'''
        final_rot_matrix = final_rot_mat(self.n_qubits, self.weight[:3])
        vector = torch.tensordot(vector, final_rot_matrix, dims=([-1],[-1]))
        #vector = torch.utils.checkpoint.checkpoint(torch.tensordot, vector, final_rot_matrix, ([-1], [-1]))
        '''Measurement'''
        # The expectation value of <vector|Z0|vector> is 1*sum_of_mag_for_the_first_half_of_amplitudes + 0*sum_of_mag_for_the_second_half_of_amplitudes
        #vector[:, 2**(self.n_qubits-1):] = 0
        #return torch.sum(torch.square(vector.abs()), -1)
        #return torch.sum(torch.square(vector[:, 2**(self.n_qubits-1):].abs()), -1) - torch.sum(torch.square(vector[:, :2**(self.n_qubits-1)].abs()), -1)
        #print(vector.square().sum()/inputs.size(0))
        return torch.tensordot(vector.abs().square(), self.parity_vector.to(inputs.device), dims=([-1],[-1]))