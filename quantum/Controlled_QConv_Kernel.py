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
    def __init__(self, channels, kernel, channel_ancillas=1, kernel_ancillas=1):
        super().__init__()
        self.channel_qubits = int(np.log2(channels))
        self.kernel_qubits = int(np.log2(kernel))
        self.channel_ancillas = channel_ancillas
        self.kernel_ancillas = kernel_ancillas
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        n_batch = inputs.size(0)
        if int(np.log2(inputs.size(1))) != self.channel_qubits + self.kernel_qubits:
            print('The input size must be compatible with the channel and kernel size. Channel and kernel size must also be powers of two.')
            quit()
        # Calculate the magnitude of inputs
        mag = inputs.square().sum(1).sqrt().reshape(-1,1)+0.0000001
        print(mag)
        # Turn the inputs into a normalized vector
        vector = torch.div(inputs, mag)
        print(vector.square().sum(1))
        # Add kernel ancilla qubits
        vector = vector.repeat_interleave(2**self.kernel_ancillas, dim=1)/np.sqrt(2**self.kernel_ancillas)
        print(vector)
        # Add channel ancilla qubits
        vector = vector.reshape(-1,2**(self.kernel_qubits+self.kernel_ancillas)).unsqueeze(1).repeat_interleave(2**self.channel_ancillas,dim=1).reshape(n_batch, -1)/np.sqrt(2**self.channel_ancillas)
        print(vector)
        return mag.reshape(-1), vector.cfloat()

'''Given concatenated single qubit matrices, return the tensor product'''
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

'''Single qubit y rotation matrix'''
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

'''Takes a channel qubit only matrix and a kernel qubit only matrix, and returns the tensor product.'''
def channel_kernel_mat_tensor(channel_mat, kernel_mat):
    size = channel_mat.size(0) * kernel_mat.size(0)
    return torch.tensordot(channel_mat.unsqueeze(-1), kernel_mat.unsqueeze(-1), dims=([-1],[-1])).permute(0,2,1,3).reshape(size,-1)

'''Rotation of kernel qubits controlled by ALL channel qubits'''
def kernel_rot_mat(channel_qubits: int, weights: nn.Parameter) -> torch.Tensor:
    kernel_qubits = weights.shape[0]
    # Computing individual kernel qubit rotation matrix
    futures = [torch.jit.fork(ry_mat, weights[qubit]*2*pi) for qubit in range(kernel_qubits)]
    results = [torch.jit.wait(fut).unsqueeze(0) for fut in futures]
    mats = torch.cat(results, 0)
    # Rotation matrix for the kernel qubits only
    rot_matrix = mat_tensor_product(mats)
    # Adds control channel qubits
    return torch.block_diag( torch.eye((2**channel_qubits - 1)*(2**kernel_qubits)).to(weights.device), rot_matrix )

'''Rotation of channel qubits independent of kernel qubits'''
def channel_rot_mat(kernel_qubits: int, weights: nn.Parameter) -> torch.Tensor:
    channel_qubits = weights.shape[0]
    # Computing individual channel qubit rotation matrix
    futures = [torch.jit.fork(ry_mat, weights[qubit]*2*pi) for qubit in range(channel_qubits)]
    results = [torch.jit.wait(fut).unsqueeze(0) for fut in futures]
    mats = torch.cat(results, 0)
    # Rotation matrix for the control qubits only
    rot_matrix = mat_tensor_product(mats)
    # Adding independent kernel qubits
    channel_mat = rot_matrix
    kernel_mat = torch.eye(2**kernel_qubits).cfloat().to(weights.device)
    return channel_kernel_mat_tensor(channel_mat, kernel_mat)
    #return torch.tensordot(rot_matrix.unsqueeze(-1), torch.eye(2**kernel_qubits).cfloat().unsqueeze(-1), dims=([-1],[-1])).permute(0,2,1,3).reshape(2**(kernel_qubits+channel_qubits),-1)

'''Vector used to compute the expectation value of parity of the output vectors. Compute by taking the dot product between the parity vector and the state vector. Row of the parity vector that corresponds to even parity has value 1 and odd parity has value -1'''
def parity_vec(n_qubits) -> torch.Tensor:
    vector = torch.ones(2**n_qubits)
    for row in torch.arange(2**n_qubits):
        row_bin = dec2bin(n_qubits, row)
        vector[row] = vector[row] - 2*(row_bin.sum()%2) # Parity
    return vector

'''Quantum convolution kernel'''
class ConvKernel(nn.Module):
    def __init__(self, channel_entangle_matrices, kernel_entangle_matrices, channels, kernel, channel_ancillas=1, kernel_ancillas=1, layers=2):
        super().__init__()
        # Determine how to embed the angles
        self.layers = layers
        self.channel_entangle_matrices = channel_entangle_matrices
        self.kernel_entangle_matrices = kernel_entangle_matrices
        self.channel_qubits = int(np.log2(channels)) + channel_ancillas
        self.kernel_qubits = int(np.log2(kernel)) + kernel_ancillas
        # Rotation angles for the channel qubits (kernel_qubits)
        self.channel_angles = nn.Parameter(torch.randn(self.layers, self.channel_qubits, dtype=torch.float32))
        self.kernel_angles = nn.Parameter(torch.randn(self.layers, self.kernel_qubits, dtype=torch.float32))
        self.encoding = Encoder(channels=channels, kernel=kernel, channel_ancillas=channel_ancillas, kernel_ancillas=kernel_ancillas)
        self.parity_vector = parity_vec(self.channel_qubits + self.kernel_qubits)
    def forward(self, inputs):
        '''Construct the variational circuit'''
        device = str(inputs.device)
        # Calculate all rotation matrices
        channel_rot_matrices_futures = [torch.jit.fork(channel_rot_mat, self.kernel_qubits, self.channel_angles[layer]) for layer in range(self.layers)]
        channel_rot_matrices = [torch.jit.wait(fut) for fut in channel_rot_matrices_futures]
        kernel_rot_matrices_futures = [torch.jit.fork(kernel_rot_mat, self.channel_qubits, self.kernel_angles[layer]) for layer in range(self.layers)]
        kernel_rot_matrices = [torch.jit.wait(fut) for fut in kernel_rot_matrices_futures]
        # Construct the sequence of matrices
        if device == 'cpu':
            matrices = [self.channel_entangle_matrices]
            for layer in range(self.layers):
                matrices.append(channel_rot_matrices[layer])
                matrices.append(self.channel_entangle_matrices)
                matrices.append(self.kernel_entangle_matrices)
                matrices.append(kernel_rot_matrices[layer])
            matrices.append(self.kernel_entangle_matrices)
        else:
            matrices = [self.channel_entangle_matrices[int(device[-1])]]
            for layer in range(self.layers):
                matrices.append(channel_rot_matrices[layer])
                matrices.append(self.channel_entangle_matrices[int(device[-1])])
                matrices.append(self.kernel_entangle_matrices[int(device[-1])])
                matrices.append(kernel_rot_matrices[layer])
            matrices.append(self.kernel_entangle_matrices[int(device[-1])])
        # Multiply all the matrices together
        matrix = torch.linalg.multi_dot(matrices)
        '''Execute the circuit'''
        # Calculate amplitude encoded states
        mag, vector = self.encoding(inputs) # vector.shape (n_batch, n_qubits)
        # Calculate the vector after the circuit (matrix)
        vector = torch.tensordot(vector, matrix, dims=([-1],[-1]))
        # Parity Measurement
        val = torch.tensordot(vector.abs().square(), self.parity_vector.to(inputs.device), dims=([-1],[-1]))
        #print('vvvvvveeeeeecccccc', vector ,'mmmmmaaaaaagggggg: ', mag, 'vvvvvvaaaaaallllll: ', val, 'ooooouuuuuutttttt: ', torch.mul(mag,val))
        return torch.mul(val, mag)