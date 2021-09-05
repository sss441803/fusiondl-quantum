import numpy as np

def calc_param(tcn_type,h,l,k):
    if tcn_type in ['DTCN', 'QTCN', 'Relu_DTCN']:
        data_size = h * k
        data_qubits = int(np.log2(data_size))
        n_qubits = data_qubits + 2
        rot_params = n_qubits * 2
        channel_params = rot_params + 3
        layer_params = h * channel_params
        tot_params = l * layer_params * 2
    if tcn_type in ['MTCN']:
        data_size = h * k
        data_qubits = int(np.log2(data_size))
        n_qubits = data_qubits + 2
        rot_params = n_qubits * 4
        channel_params = rot_params + 3
        layer_params = h * channel_params
        tot_params = l * layer_params * 2
    if tcn_type in ['TCN', 'SecondTCN', 'smallTCN']:
        channel_params = h * k
        layer_params = h * channel_params
        tot_params = l * layer_params * 2
    return tot_params

tcn_type='DTCN'
tcn_hidden = [2,4,8]
tcn_layers = [2,4,6,8]
tcn_kernel = [4,8]
ancillas = 1
ROCs = np.zeros((len(tcn_hidden)*len(tcn_layers)*len(tcn_kernel), 30))
num_params = np.zeros(len(tcn_hidden)*len(tcn_layers)*len(tcn_kernel))
largest_ROC = 0
for h, hidden in enumerate(tcn_hidden):
    for l, layers in enumerate(tcn_layers):
        for k, kernel in enumerate(tcn_kernel):
            try:
                file1 = open('outputs/{}_h{}l{}k{}a{}.out'.format(tcn_type,hidden,layers,kernel,ancillas), 'r')
            except:
                print('file \'outputs/{}_h{}l{}k{}a{}.out\' doesn\'t exist'.format(tcn_type,hidden,layers,kernel,ancillas))
                continue
            file_num = h*len(tcn_layers)*len(tcn_kernel)+l*len(tcn_kernel)+k
            num_params[file_num] = calc_param(tcn_type,hidden,layers,kernel)
            n = 0
            for line in file1:
                if 'Validation ROC' in line:
                    ROC = float(line[16:-1])
                    ROCs[file_num, n] = ROC
                    if ROC > largest_ROC:
                        largest_ROC = ROC
                        largest_ROC_pos = 'h{}l{}k{}'.format(hidden,layers,kernel)
                    n = n+1
                #if 'Training Loss numpy' in line:
                    #print(line[16:-1])
                    #print(line[21:-1])
            
np.savetxt('{}.csv'.format(tcn_type), ROCs, delimiter=',')

print('Row containing max: ', ROCs.argmax()//30, ' corresponding to file ', largest_ROC_pos, '. Column containing max: ', ROCs.argmax()%30, '. Max value: ', ROCs.max())

max_ROCs = ROCs.max(1)
print('Number of parameters vs ROC: ')
for file_num in range(len(tcn_hidden)*len(tcn_layers)*len(tcn_kernel)):
    print(int(num_params[file_num]), '     ', max_ROCs[file_num])