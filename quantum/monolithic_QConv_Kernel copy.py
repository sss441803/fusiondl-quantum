import torch
import torch.nn as nn
import numpy as np

def dec2bin(n_bits, x):
    device = x.device
    mask = 2 ** torch.arange(n_bits - 1, -1, -1).to(device)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0)

def bin2dec(n_bits, b):
    device = b.device
    mask = 2 ** torch.arange(n_bits - 1, -1, -1).to(device)
    return torch.sum(mask * b, -1)

'''Creates the encoded n qubit state vector based on variation encoding'''
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, ry_angles: torch.Tensor, rz_angles: torch.Tensor) -> torch.Tensor:
        device = ry_angles.device
        ry_angles = ry_angles/2
        rz_angles = rz_angles/2
        '''Parallel compute entries for the ry matrix'''
        ry_entry11 = torch.cos(ry_angles).unsqueeze(-1)
        ry_entry12 = -torch.sin(ry_angles).unsqueeze(-1)
        ry_entry21 = torch.sin(ry_angles).unsqueeze(-1)
        ry_entry22 = torch.cos(ry_angles).unsqueeze(-1)

        '''Parallel compute entries for the rz matrix'''
        rz_entry11 = torch.exp(-1j*rz_angles).unsqueeze(-1)
        #rz_entry12 = torch.zeros(n_batch, n_qubits).unsqueeze(-1)
        #rz_entry21 = torch.zeros(n_batch, n_qubits).unsqueeze(-1)
        rz_entry22 = torch.exp(1j*rz_angles).unsqueeze(-1)
        '''Parallel compute entries for the composite rz*ry matrix'''
        entry11 = ry_entry11 * rz_entry11
        entry12 = ry_entry12 * rz_entry11
        entry21 = ry_entry21 * rz_entry22
        entry22 = ry_entry22 * rz_entry22

        mat = torch.cat((entry11, entry12, entry21, entry22), -1) # mat.shape (n_batch, n_qubits, 4)

        n_batch, n_qubits = torch.tensor(mat.size(0)), torch.tensor(mat.size(1))
        #tensor_mat = torch.zeros(n_batch, 2**n_qubits, 2**n_qubits, dtype=torch.cfloat)
        '''Since the initial state is |00...000>, i.e. (1,0,0,...), the first column of the rz*ry matrix is the result'''
        for row in torch.arange(2**n_qubits):
            row_bin = dec2bin(n_qubits, row) # Returns a tensor of bool that stands for the basis for the nth qubit
            #print('row_bin ', row_bin)
            #for col in range(2**n_qubits):
            #    col_bin = bin(col)[2:]
            #    col_bin = '0'*(n_qubits-len(col_bin))+col_bin
                #print('col_bin ', col_bin)
            #    entries = torch.cat([mat[:, qubit, 2*int(row_bin[qubit])+int(col_bin[qubit])].unsqueeze(1) for qubit in range(n_qubits)], 1)
            #    tensor_mat[:, row, col] = torch.prod(entries, dim=1)

            col = torch.tensor(0)
            col_bin = dec2bin(n_qubits, col)
            #print('col_bin ', col_bin)
            val_list = [mat[:, qubit, 2*row_bin[qubit]+col_bin[qubit]].unsqueeze(1) for qubit in range(n_qubits)] # List of values that needs to be multiplied to obtain the row_th row of the resulting state vector. 2*int(row_bin[qubit])+int(col_bin[qubit]) computes which entry of the 2 by 2 matrix for this qubit should be chosen.
            entries = torch.cat(val_list, 1)
            if row == 0:
                vector = torch.prod(entries, dim=1).unsqueeze(-1)
            else:
                vector = torch.cat((vector, torch.prod(entries, dim=1).unsqueeze(-1)), 1)
        return vector

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
    rx_matrix_list = ry_mat_list(thetas[0])
    ry_matrix_list = ry_mat_list(thetas[1])
    rz_matrix_list = ry_mat_list(thetas[2])
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
    mats = torch.cat([torch.matmul(rx_mat(weights[qubit]), rz_mat(weights[n_qubits+qubit])).unsqueeze(0) for qubit in range(n_qubits)], 0)
    rot_matrix = mat_tensor_product(mats)
    return rot_matrix

class ConvKernel(nn.Module):
    def __init__(self, entangle_matrix, n_qubits, initial=False):
        super().__init__()
        # Determine how to embed the angles
        self.initial = initial
        self.n_qubits = n_qubits
        self.encoding = Encoder()
        self.weight = nn.Parameter(torch.randn(3+2*n_qubits, dtype=torch.float32))
        self.entangle_matrix = entangle_matrix
    def forward(self, inputs):
        device = str(inputs.device)
        # inputs.shape (n_batch, n_qubits)
        '''Calculate angles of rotation for variational encoding'''
        if self.initial:
            ry_angles = torch.arctan(inputs/5)
            rz_angles = torch.arctan(torch.square(inputs/5))
        else:
            ry_angles = inputs
            rz_angles = torch.square(inputs)
        '''Calculate rotated states'''
        vector = self.encoding(ry_angles, rz_angles) # vector.shape (n_batch, 2**n_qubits)
        #vector = self.encoding(inputs, torch.square(inputs))
        #entangle_matrix = torch.Variable(entangle_mat(self.n_qubits, self.device).to(self.device))
        #print('vector device ', vector.device, ' entangle_matrix device ', self.entangle_matrix.device)
        '''Product of entangle matrix and rotation matrix'''
        rot_matrix = rot_mat(self.weight[3:])
        if device == 'cpu':
            matrix = torch.chain_matmul(self.entangle_matrix, rot_matrix, self.entangle_matrix)
        else:
            matrix = torch.chain_matmul(self.entangle_matrix[int(device[-1])], rot_matrix, self.entangle_matrix[int(device[-1])])
        '''Rotation then Entangle'''
        vector = torch.tensordot(vector, matrix, dims=([-1],[-1]))
        '''Final rotation'''
        final_rot_matrix = final_rot_mat(self.n_qubits, self.weight[:3])
        vector = torch.tensordot(vector, final_rot_matrix, dims=([-1],[-1]))
        '''Measurement'''
        # The expectation value of <vector|Z0|vector> is 1*sum_of_mag_for_the_first_half_of_amplitudes + 0*sum_of_mag_for_the_second_half_of_amplitudes
        vector[:, 2**(self.n_qubits-1):] = 0
        return torch.sum(torch.square(vector.abs()), -1)