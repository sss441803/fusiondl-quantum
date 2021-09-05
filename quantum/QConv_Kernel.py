import torch
import torch.nn as nn
import torch.utils.checkpoint
import numpy as np
import math
from typing import List

pi = np.pi

def dec2bin(n_bits, x):
    device = x.device
    mask = 2 ** torch.arange(n_bits - 1, -1, -1).to(device)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0)

def bin2dec(n_bits, b):
    device = b.device
    mask = 2 ** torch.arange(n_bits - 1, -1, -1).to(device)
    return torch.sum(mask * b, -1)

#'''Creates the encoded n qubit state vector based on variation encoding'''
#'''Follows arxiv:2011.00027 fig 4. First Hadamard, then Rz(x)'''
#class Encoder(nn.Module):
#    def __init__(self):
#        super().__init__()
#    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
#        angles = 2*pi*inputs
#        '''Parallel compute the state vector values for all qubits for the whole batch'''
#        v_vals = torch.cat( [torch.exp(-1j*angles/2).unsqueeze(-1), torch.exp(1j*angles/2).unsqueeze(-1)], dim=-1)/np.sqrt(2) # First is amplitude for |0>, second is for |1>
#
#        n_qubits = torch.tensor(angles.size(1))
#
#        for row in torch.arange(2**n_qubits):
#            row_bin = dec2bin(n_qubits, row) # Returns a tensor of bool that stands for the basis for the nth qubit
#            val_list = [v_vals[:, qubit, int(row_bin[qubit])].unsqueeze(1) for qubit in torch.arange(n_qubits)] # List of values that needs to be multiplied to obtain the row_th row of the resulting state vector.
#            entries = torch.cat(val_list, 1)
#            if row == 0:
#                vector = torch.prod(entries, dim=1).unsqueeze(-1)
#            else:
#                vector = torch.cat((vector, torch.prod(entries, dim=1).unsqueeze(-1)), 1)
#        return vector

'''Creates the encoded n qubit state vector based on variation encoding.'''
'''Follows arxiv:2011.00027 fig 4. First Hadamard, then Rz(x).'''
def Encoder(inputs: torch.Tensor) -> torch.Tensor:
    angles = 2*pi*inputs
    '''Parallel compute the state vector values for all qubits for the whole batch'''
    v_vals = torch.cat( [torch.exp(-1j*angles/2).unsqueeze(-1), torch.exp(1j*angles/2).unsqueeze(-1)], dim=-1)/np.sqrt(2) # First is amplitude for |0>, second is for |1>
    n_qubits = torch.tensor(angles.size(1))

    '''Parallel compute each entry of the output vector. See https://pytorch.org/tutorials/advanced/torch-script-parallelism.html.'''
    '''Each entry of futures is a row of the final vector'''
    futures : List[torch.jit.Future[torch.Tensor]] = []
    for row in torch.arange(2**n_qubits):
        futures.append(torch.jit.fork(enc_vector_entry_calc, row, n_qubits, v_vals))
    
    results = []
    for future in futures:
        results.append(torch.jit.wait(future))
    
    '''Concatenating entries to form a vector'''
    return torch.cat(results, 1)

'''Calculates one entry of the vector for the encoder. This function will be applied in parallel in Encoder.'''
def enc_vector_entry_calc(row, n_qubits, v_vals):
    row_bin = dec2bin(n_qubits, row) # Returns a tensor of bool that stands for the basis for the nth qubit
    val_list = [v_vals[:, qubit, int(row_bin[qubit])].unsqueeze(1) for qubit in torch.arange(n_qubits)] # List of values that needs to be multiplied to obtain the row_th row of the resulting state vector.
    entries = torch.cat(val_list, 1)
    vector_entry = torch.prod(entries, dim=1).unsqueeze(-1)
    return vector_entry

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

'''Execute single qubit rotation on a tensor product state. This function is called in higher_order_encoding.'''
def single_qubit_z_rot(states, angles, qubit) -> torch.Tensor:
    rows = states.size(1)
    n_qubits = int(math.log2(rows))
    '''Each entry of futures is a row of the final vector'''
    futures = [torch.jit.fork(z_rot_vector_entry_calc, row, n_qubits, states, angles, qubit) for row in torch.arange(rows)]
    results = [torch.jit.wait(fut) for fut in futures]
    return torch.cat(results, -1)

'''Calculates one entry of the vector for the single_qubit_z_rot operation. This function will be applied in parallel in single_qubit_z_rot.'''
def z_rot_vector_entry_calc(row, n_qubits, states, angles, qubit):
    row_bin = dec2bin(n_qubits, row)
    if row_bin[qubit]:
        return (torch.exp(-1j*angles/2)*states[:,row]).unsqueeze(-1)
    else:
        return (torch.exp(1j*angles/2)*states[:,row]).unsqueeze(-1)

'''Constructs a column vector for a CNOT matrix. This function is called in CNOT_mat.'''
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

'''Constructs a CNOT matrix with designated control and target qubits. This function is called in entangle_mat and higher_order_encoding.'''
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
    n_qubits = weights.shape[0]//2
    mats = torch.cat([torch.matmul(rx_mat(weights[qubit]*2*pi), rz_mat(weights[n_qubits+qubit]*2*pi)).unsqueeze(0) for qubit in range(n_qubits)], 0)
    rot_matrix = mat_tensor_product(mats)
    return rot_matrix

class ConvKernel(nn.Module):
    def __init__(self, entangle_matrix, n_qubits):
        super().__init__()
        # Determine how to embed the angles
        self.n_qubits = n_qubits
        #self.encoding = Encoder()
        self.encoding = Encoder
        self.weight = nn.Parameter(torch.randn(3+2*n_qubits, dtype=torch.float32))
        self.entangle_matrix = entangle_matrix
    def forward(self, inputs):
        device = str(inputs.device)
        if inputs.max() > 1:
            inputs = torch.arctan(inputs)
        '''Calculate rotated states'''
        vector = self.encoding(inputs) # vector.shape (n_batch, 2**n_qubits)
        '''Further higher order encoding'''
        vector = higher_order_encoding(vector, inputs)
        '''Product of entangle matrix and rotation matrix'''
        rot_matrix = rot_mat(self.weight[3:])
        if device == 'cpu':
            matrix = torch.chain_matmul(self.entangle_matrix, rot_matrix, self.entangle_matrix)
        else:
            matrix = torch.chain_matmul(self.entangle_matrix[int(device[-1])], rot_matrix, self.entangle_matrix[int(device[-1])])
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
        return torch.sum(torch.square(vector[:, 2**(self.n_qubits-1):].abs()), -1) - torch.sum(torch.square(vector[:, :2**(self.n_qubits-1)].abs()), -1)