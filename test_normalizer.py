import numpy as np



path = './normalization/normalization_signal_group_195062205603422687461982563883491993390.npz' 

dat = np.load(path, encoding="latin1", allow_pickle=True)



print(dat)

for k in dat:
    print(k)

stds = (dat['stds'][()])
 
for m in stds:
    print(m)
    print(stds[m].shape)
    median = np.median (stds[m],axis = 0)
    print(median.shape)
    print(median)


