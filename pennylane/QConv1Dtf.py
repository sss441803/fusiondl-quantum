import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pennylane as qml

import time

# Need decompose a multichannel quantum convolution layer into many single channel quantum convolution. This is because each torch.nn.Module converted from a qml.qnode must be associated with a device that contains all the needed qubits. Having all the qubits in one layer is too many for one device.

'''
# This kernel is for classical CNN and can be used inplace of the quantum kernel
class kernel_module(nn.Module):
    def __init__(self, kernel_size):
        super(kernel_module, self).__init__()
        self.kernel_size = kernel_size
        self.kernel = torch.randn(kernel_size)
        self.flat_kernel = self.kernel.reshape(-1)
    def forward(self, x):
        out = torch.tensordot(x, self.flat_kernel, dims=([-1],[0]))
        return out'''

# defines the quantum convolutional kernel (a variational circuit)
def build_qconv(dev, kernel_size):
    n_qubits = kernel_size
    # Defines the q_node that is the quantum convolution kernel
    @qml.qnode(dev, interface='tf', diff_method='backprop')
    def circuit(inputs, weights):
            qml.templates.AngleEmbedding(tf.math.atan(inputs), rotation='Y', wires=range(n_qubits))
            qml.templates.AngleEmbedding(tf.math.atan(inputs*inputs), rotation='Z', wires=range(n_qubits))
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
    q_kernel = qml.qnn.KerasLayer(circuit, weight_shapes, output_dim=n_qubits)
    return q_kernel

# Initialize convolutional layer with the custom kernel
class CustomKernel_1In1Out_Conv1D(keras.layers.Layer):
    def __init__(self, kernel, kernel_size, strides=1, padding='valid'):
        # x.shape (n_batch, length_of_profile)
        self.kernel = kernel
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding.upper()
    def __call__(self, inputs):
        n_batch = inputs.shape[0]
        inputs = tf.reshape(inputs, [n_batch, 1, -1, 1]) # x.shape (n_batch, 1, length_of_profile, 1)
        inputs = tf.image.extract_patches(inputs, sizes=[1, 1, self.kernel_size, 1], strides=[1, 1, self.strides, 1], rates=[1, 1, 1, 1], padding=self.padding) # x.shape (n_batch, 1, out_length, kernel_size)
        inputs = tf.reshape(inputs, [n_batch, -1, self.kernel_size]) # x.shape (n_batch, out_length, kernel_size)
        out = self.kernel(inputs) # out.shape (n_batch, out_length)
        return out

# 1In1Out channel quantum convolution 2D
class Q_1In1Out_Conv1D(keras.layers.Layer):
    def __init__(self, dev, kernel_size, strides=1, padding='valid'):
        q_kernel = build_qconv(dev, kernel_size)
        self.conv = CustomKernel_1In1Out_Conv1D(q_kernel, kernel_size, strides=strides, padding=padding)
    def __call__(self, inputs):
        out = self.conv(inputs)
        return out

# Multichannel input one channel output quantum convolution 2D
class Q_MulIn1Out_Conv1D(keras.layers.Layer):
    def __init__(self, dev, in_channels, kernel_size, strides=1, padding='valid'):
        self.in_channels = in_channels
        self.convs = []
        for _ in range(in_channels):
            self.convs.append(Q_1In1Out_Conv1D(dev, kernel_size, strides))
    def __call__(self, inputs):
        # inputs.shape (n_batch, in_channels, length_of_profile)
        for channel in range(self.in_channels):
            x_channel = inputs[:, channel] # x_channel.shape (n_batch, length_of_profile)
            conv = self.convs[channel]
            out = conv(x_channel) # out.shape (n_batch, out_length)
            out = tf.expand_dims(out, 1) # out.shape (n_batch, 1, out_length)
            output = out if channel==0 else tf.concat([output, out], 1)
            print('in_channel: ', channel)
        # output.shape (n_batch, n_channels, out_length)
        output = layers.Dense(1)(tf.transpose(output, perm=[0,2,1])) # output.shape (n_batch, out_length, 1)
        output = tf.squeeze(output, -1) # output.shape (n_batch, out_length)
        return output

# Quantum convolution 2D layer
class QConv1D(keras.layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, strides=1, padding='valid'):
        self.in_channels = in_channels
        self.out_channels = out_channels
        n_qubits = kernel_size
        dev = qml.device("default.qubit.tf", wires=n_qubits)
        self.convs = []
        for _ in range(self.out_channels):
            self.convs.append(Q_MulIn1Out_Conv1D(dev, in_channels, kernel_size, strides, padding))
    def __call__(self, inputs):
        # inputs.shape (n_batch, in_channels, length_of_profile)
        for channel in range(self.out_channels):
            conv = self.convs[channel]
            out = conv(x) # out.shape (n_batch, out_length)
            out = tf.expand_dims(out, 1) # out.shape (n_batch, 1, out_length)
            output = out if channel==0 else tf.concat([output, out], 1)
            print('out_channel: ', channel)
        # output.shape (n_batch, out_channels, out_length)
        return output
'''
# Testing code
'''
'''
tf.debugging.set_log_device_placement(True)

# Create some tensors
a = tf.random.uniform([1000,2000])
b = tf.random.uniform([2000,1000])

for _ in range(10):
    c = tf.matmul(a, b)
    print(c)
    time.sleep(2)'''

print('now pennylane')

kernel_size = 5
n_qubits = kernel_size
'''
dev = qml.device("default.qubit.tf", wires=n_qubits) 
#qk = build_qconv(dev, kernel_size)

x = tf.random.uniform((9,5), )
out = qk(x)
print(out)
x = tf.random.uniform((9,5,3))
out = qk(x)
print(out)
'''
for _ in range(1):
    x = tf.random.uniform((5,10,100))
    start = time.time()
    #out = CustomKernel_1In1Out_Conv1D(x, qk, kernel_size, strides=1, padding='valid')
    out=QConv1D(10, 5, kernel_size, strides=1, padding='valid')(x)
    stop = time.time()
    print('Successfully computed the output with shape: ', out.shape, '. Total time elapsed: ', stop - start, ' seconds.')

'''
dev = qml.device("default.qubit.tf", wires=1)
@qml.qnode(dev, interface="tf", diff_method="backprop")
def circuit(inputs):
    qml.RX(inputs[1], wires=0)
    qml.Rot(inputs[0], inputs[1], inputs[2], wires=0)
    return qml.expval(qml.PauliZ(0))


weight_shapes = {} # weights are the rotation angles for each qubit. There are 3 angles to fully specify the rotation
    # Make the kernel a torch.nn.Module object
q_kernel = qml.qnn.KerasLayer(circuit, weight_shapes, output_dim=1)

weights = tf.Variable([0.2, 0.5, 0.1])
print(circuit(weights))
print(q_kernel(weights))'''