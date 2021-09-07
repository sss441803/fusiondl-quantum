from loader import Loader
from preprocess import guarantee_preprocessed
from pprint import pprint
from conf import conf
import numpy as np
import torch
'''
#########################################################
This file trains a deep learning model to predict
disruptions on time series data from plasma discharges.

Dependencies:
conf.py: configuration of model,training,paths, and data
builder.py: logic to construct the ML architecture
data_processing.py: classes to handle data processing

Author: Julian Kates-Harbeck, jkatesharbeck@g.harvard.edu

This work was supported by the DOE CSGF program.
#########################################################
'''

import datetime
import random
import sys
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--kernel_spatial', type=int, help='Kernel size of spatial convolution.', default=4)
parser.add_argument('--channels_spatial', nargs="+", help='List of spatial convolution output channel sizes. Format should be \'c4\' \'q3\'...\'e2\' classical convolution with 4 output channels, quantum with 3 channels ... and easy quantum with 2 channels.', default=['c16', 'c8', 'c4', 'c2'])
parser.add_argument('--linear_sizes', type=int, nargs="+", help='List of linear layer output sizes after CNN before TCN. The last layer is input size for TCN.', default=[20, 5])
parser.add_argument('--kernel_temporal', type=int, help='Kernel size of temporal convolution.', default=5)
parser.add_argument('--channels_temporal', nargs="+", help='List of temporal convolution output channel sizes. Format should be \'c4\' \'q3\' ... \'e2\' classical convolution with 4 output channels, quantum with 3 channels ... and easy quantum with 2 channels.', default=['c5','c5','c5','c5'])
parser.add_argument('--no_scalars', type=bool, help='If Ture, then no 0D scalar signals will be used.', default=False)
parser.add_argument('--input_div', type=float, help='Factor by which the inputs are divided.', default=1.0)
parser.add_argument('--subsampling', type=int, help='Input data timestep Subsampling factor.', default=1)
parser.add_argument('--tcn_type', type=str, help='Type of tcn. choose from c (classical), e (easy quantum), q (quantum), d (dense quantum), m (more parameter dense quantum) or t (controlled quantum). If this is used, it overrides the channels_temporal argument. Must be used in conjunction with tcn_hidden and tcn_layers.', default='c')
parser.add_argument('--tcn_hidden', type=int, help='Number of channels per tcn layer. If this is used, it overrides the channels_temporal argument. Must be used in conjunction with tcn_layers and tcn_type.', default=0)
parser.add_argument('--tcn_layers', type=int, help='Number of tcn layers. If this is used, it overrides the channels_temporal argument. Must be used in conjunction with tcn_hidden and tcn type.', default=0)
parser.add_argument('--ancillas', type=int, help='Number of ancilla qubits to use if using dense quantum convolution.', default=2)
parser.add_argument('--threads', type=int, help='Number of threads.', default=10)
args = parser.parse_args()
torch.set_num_threads(args.threads)
if args.tcn_hidden != 0:
    args.channels_temporal = ['c' + str(args.tcn_hidden)]
    args.channels_temporal = args.channels_temporal + [args.tcn_type + str(args.tcn_hidden)]*args.tcn_layers
print('Arguments: ', args)

#pprint(conf)

if 'torch' in conf['model'].keys() and conf['model']['torch']:
    from quantum.runner import (
        train, make_predictions_and_evaluate_gpu
        )

if conf['data']['normalizer'] == 'minmax':
    from normalize import MinMaxNormalizer as Normalizer
elif conf['data']['normalizer'] == 'meanvar':
    from normalize import MeanVarNormalizer as Normalizer
elif conf['data']['normalizer'] == 'var':
    # performs !much better than minmaxnormalizer
    from normalize import VarNormalizer as Normalizer
elif conf['data']['normalizer'] == 'averagevar':
    # performs !much better than minmaxnormalizer
    from normalize import (
        AveragingVarNormalizer as Normalizer
    )
else:
    print('unkown normalizer. exiting')
    exit(1)

shot_list_dir = conf['paths']['shot_list_dir']
shot_files = conf['paths']['shot_files']
shot_files_test = conf['paths']['shot_files_test']
train_frac = conf['training']['train_frac']
stateful = conf['model']['stateful']

np.random.seed(0)
random.seed(0)

#only_predict = len(sys.argv) > 1
only_predict = False
custom_path = None
if only_predict:
    custom_path = sys.argv[1]
    print("predicting using path {}".format(custom_path))

#####################################################
#                   PREPROCESSING                   #
#####################################################
# TODO(KGF): check tuple unpack
(shot_list_train, shot_list_validate,
 shot_list_test) = guarantee_preprocessed(conf)

#####################################################
#                   NORMALIZATION                   #
#####################################################

nn = Normalizer(conf)
nn.train()
loader = Loader(conf, nn)
print("...done")
print('Training on {} shots, testing on {} shots'.format(
    len(shot_list_train), len(shot_list_test)))


#####################################################
#                    TRAINING                       #
#####################################################
if not only_predict:
    train(conf, shot_list_train, shot_list_validate, loader, args)#, shot_list_test)

#####################################################
#                    PREDICTING                     #
#####################################################
loader.set_inference_mode(True)

# load last model for testing
print('saving results')
y_prime = []
y_prime_test = []
y_prime_train = []

y_gold = []
y_gold_test = []
y_gold_train = []

disruptive = []
disruptive_train = []
disruptive_test = []

# y_prime_train, y_gold_train, disruptive_train =
#         make_predictions(conf, shot_list_train, loader)
# y_prime_test, y_gold_test, disruptive_test =
#         make_predictions(conf, shot_list_test, loader)

# TODO(KGF): check tuple unpack
device=torch.device('cuda')
(y_prime_train, y_gold_train, disruptive_train, roc_train,
 loss_train) = make_predictions_and_evaluate_gpu(
     conf, shot_list_train, loader, custom_path,device=device)
(y_prime_test, y_gold_test, disruptive_test, roc_test,
 loss_test) = make_predictions_and_evaluate_gpu(
     conf, shot_list_test, loader, custom_path,device=device)
print('=========Summary========')
print('Train Loss: {:.3e}'.format(loss_train))
print('Train ROC: {:.4f}'.format(roc_train))
print('Test Loss: {:.3e}'.format(loss_test))
print('Test ROC: {:.4f}'.format(roc_test))


disruptive_train = np.array(disruptive_train)
disruptive_test = np.array(disruptive_test)

y_gold = y_gold_train + y_gold_test
y_prime = y_prime_train + y_prime_test
disruptive = np.concatenate((disruptive_train, disruptive_test))

shot_list_validate.make_light()
shot_list_test.make_light()
shot_list_train.make_light()

save_str = 'results_' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
result_base_path = conf['paths']['results_prepath']
if not os.path.exists(result_base_path):
    os.makedirs(result_base_path)
np.savez(result_base_path+save_str, y_gold=y_gold, y_gold_train=y_gold_train,
         y_gold_test=y_gold_test, y_prime=y_prime, y_prime_train=y_prime_train,
         y_prime_test=y_prime_test, disruptive=disruptive,
         disruptive_train=disruptive_train, disruptive_test=disruptive_test,
         shot_list_validate=shot_list_validate,
         shot_list_train=shot_list_train, shot_list_test=shot_list_test,
         conf=conf)

print('finished.')
