from __future__ import print_function
import random
from torchsummary import summary
import numpy as np
import sys
if sys.version_info[0] < 3:
    from itertools import imap

#leading to import errors:
#from hyperopt import hp, STATUS_OK
#from hyperas.distributions import conditional

import time
import datetime
import os
from functools import partial
import pathos.multiprocessing as mp

from conf import conf
from loader import Loader, ProcessGenerator
from performance import PerformanceAnalyzer
from evaluation import *
from downloading import makedirs_process_safe


import hashlib

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as opt
from torch.nn.utils import weight_norm
from convlstmnet import * #Conv_LSTM_net NVdia
from custom_quantum_models import * #FTCN #FLSTM

#from torch_QConv_Kernel import device

model_filename = 'torch_modelfull_model.pt'

def build_torch_model(conf):
    dropout = conf['model']['dropout_prob']
# dim = 10

    # lin = nn.Linear(input_size,intermediate_dim)
    n_scalars, n_profiles, profile_size = get_signal_dimensions(conf)
    print('n_scalars,n_profiles,profile_size=',n_scalars,n_profiles,profile_size)
    dim = n_scalars+n_profiles*profile_size
    input_size = dim
    try:
       output_size = len(conf['training']['target_description'])
    except:
        output_size = 1
    # intermediate_dim = 15

    layer_sizes_spatial = [2]
    #layer_sizes_spatial = [16,8,8]
    kernel_size_spatial= 3
    linear_size = 10
    linear_layer_num = 2
    
    try:
      num_channels_tcn = [conf['model']['tcn_hidden']]*conf['model']['tcn_layers']#[3]*5
      kernel_size_temporal = conf['model']['kernel_size_temporal'] #3
    except:
      num_channels_tcn =  [40]*5 # [conf['model']['tcn_hidden']]*conf['model']['tcn_layers']#[3]*5
      kernel_size_temporal = 3 #conf['model']['kernel_size_temporal'] #3
    try:
      model_type = conf['model']['model_type']
    except:
      model_type='LSTM'
    
    if model_type == 'TCN':  
        model = FTCN(n_scalars,n_profiles,profile_size,layer_sizes_spatial,
             kernel_size_spatial,linear_size,output_size,num_channels_tcn,
             kernel_size_temporal,dropout,linear_layer_num)
    elif model_type == 'LSTM':
         rnn_size=conf['model']['rnn_size']
         rnn_layers=conf['model']['rnn_layers']
         model=FLSTM(output_dim=output_size,rnn_layers=rnn_layers,rnn_size=rnn_size,
             dropout=dropout,layer_sizes_spatial=layer_sizes_spatial,
             kernel_size_spatial=kernel_size_spatial, linear_layer_num=linear_layer_num,
             n_scalars=n_scalars,n_profiles=n_profiles,batch_first=True,bidirectional=False)
    
    elif model_type == 'TTLSTM':
        try:
          tt_dense = conf['model']['tt_lstm_hidden']
          cell_order = conf['model']['cell_order']
          cell_steps = conf['model']['cell_steps']
          cell_rank = conf['model']['cell_rank']
        except:
           tt_dense=20
           cell_order=2
           cell_steps=2
           cell_rank=2
        model = FTTLSTM (tt_dense=tt_dense,cell_steps=cell_steps,cell_rank=cell_rank,
                cell_order=cell_order,output_dim=output_size,dense_size=linear_size,dropout=dropout,
                batch_first=True,n_scalars=n_scalars,n_profiles=n_profiles,
                profile_size=profile_size,layer_sizes_spatial=layer_sizes_spatial, 
                kernel_size_spatial=kernel_size_spatial, linear_layer_num=linear_layer_num)

    else:
        print('!!!!!!!!!!!!Architecture NOT implemented.')
        exit(1)
 
    return model



def add_noise(X,conf = None):
        if conf['training']['noise']==True:
           prob=0.05
        else:
           prob=conf['training']['noise']
        for i in range(0,X.shape[0]):
            for j in range(0,X.shape[2]):
                a=random.randint(0,100)
                if a<prob*100:
                   X[i,:,j]=0.0
        return X


def get_signal_dimensions(conf):
    #make sure all 1D indices are contiguous in the end!
    use_signals = conf['paths']['use_signals']
    n_scalars = 0
    n_profiles = 0
    profile_size = 0
    is_1D_region = use_signals[0].num_channels > 1#do we have any 1D indices?
    for sig in use_signals:
        num_channels = sig.num_channels
        if num_channels > 1:
            profile_size = num_channels
            n_profiles += 1
            is_1D_region = True
        else:
            assert(not is_1D_region), "make sure all use_signals are ordered such that 1D signals come last!"
            assert(num_channels == 1)
            n_scalars += 1
            is_1D_region = False
    return n_scalars,n_profiles,profile_size 

def apply_model_to_np(model,x,device=None):
    #     return model(Variable(torch.from_numpy(x).float()).unsqueeze(0)).squeeze(0).data.numpy()
    return model(Variable(torch.from_numpy(x).float()).to(device)).to(torch.device('cpu')).data.numpy()



def make_predictions(conf,shot_list,loader,custom_path=None,inference_model=None,device=None):
    generator = loader.inference_batch_generator_full_shot(shot_list)
    if inference_model == None:
      if custom_path == None:
        model_path = get_model_path(conf)
      else:
        model_path = custom_path
      print('model-path is: ',model_path)
    #  inference_model = build_torch_model(conf)
      inference_model=torch.load(model_path)
    #  inference_model.load_state_dict(torch.load(model_path))
   #   inference_model.to(device)
    #shot_list = shot_list.random_sublist(10)
    inference_model.eval()
    y_prime = []
    y_gold = []
    disruptive = []
    num_shots = len(shot_list)

    while True:
        x,y,mask,disr,lengths,num_so_far,num_total = next(generator)
        #x, y, mask = Variable(torch.from_numpy(x_).float()), Variable(torch.from_numpy(y_).float()),Variable(torch.from_numpy(mask_).byte())
        output = apply_model_to_np(inference_model,x,device=device)
        for batch_idx in range(x.shape[0]):
            curr_length = lengths[batch_idx]
            y_prime += [output[batch_idx,:curr_length,:]]
            y_gold += [y[batch_idx,:curr_length,:]]
            disruptive += [disr[batch_idx]]
        if len(disruptive) >= num_shots:
            y_prime = y_prime[:num_shots]
            y_gold = y_gold[:num_shots]
            disruptive = disruptive[:num_shots]
            break
    return y_prime,y_gold,disruptive

def make_predictions_and_evaluate_gpu(conf,shot_list,loader,custom_path = None,inference_model=None,device=None):
    y_prime,y_gold,disruptive = make_predictions(conf,shot_list,loader,custom_path,inference_model=inference_model,device=device)
    print('y_prime,',len(y_prime),len(y_prime[0]))
    print('y_gold,',len(y_gold),len(y_gold[0]))
    print('disruptive,',len(disruptive))
    analyzer = PerformanceAnalyzer(conf=conf)
    roc_area = analyzer.get_roc_area(y_prime,y_gold,disruptive)
    #roc_area=0.0
    loss = get_loss_from_list(y_prime,y_gold,conf['data']['target'])
    return y_prime,y_gold,disruptive,roc_area,loss


def get_model_path(conf):
    return conf['paths']['model_save_path']  + model_filename #save_prepath + model_filename


def train_epoch(model,data_gen,optimizer,loss_fn,device=None,conf = {}):
    loss = 0
    total_loss = 0
    num_so_far = 0
    x_,y_,mask_,num_so_far_start,num_total = next(data_gen)
    if 'noise' in conf['training'].keys() and conf['training']['noise']!=False:
            x_=add_noise(x_,conf = conf)
    num_so_far = num_so_far_start
    step = 0
    #sampling_index = torch.arange(0, 5700, 10).to(device)
    while True:
        x, y, mask = Variable(torch.from_numpy(x_).float()).to(device), Variable(torch.from_numpy(y_).float()).to(device),Variable(torch.from_numpy(mask_).byte()).to(device).to(torch.bool)
        #x, y, mask = torch.index_select(x, 1, sampling_index), torch.index_select(y, 1, sampling_index), torch.index_select(mask, 1, sampling_index)
        optimizer.zero_grad()
        output = model(x)
        output_masked = torch.masked_select(output,mask)
        y_masked = torch.masked_select(y,mask)
    #    print('INPUTSHAPING::')
    #    print('x.shape,',x.shape)
    #    print('OUTPUTSHAPING::')
    #    print('y.shape:',y.shape)
    #    print('output.shape:',output.shape)
        loss = loss_fn(output_masked,y_masked)
        total_loss += loss.data.item()
        
        loss.backward()
        optimizer.step()
        step += 1
        print("[{}]  [{}/{}] loss: {:.3f}, ave_loss: {:.3f}".format(step,num_so_far-num_so_far_start,num_total,loss.data.item(),total_loss/step))
        if num_so_far-num_so_far_start >= num_total:
            break
        x_,y_,mask_,num_so_far,num_total = next(data_gen)
    return step,loss.data.item(),total_loss/step,num_so_far,1.0*num_so_far/num_total


def train(conf,shot_list_train,shot_list_validate,loader):

    np.random.seed(1)
    use_cuda=False
    device = torch.device("cuda")
    #data_gen = ProcessGenerator(partial(loader.training_batch_generator_full_shot_partial_reset,shot_list=shot_list_train)())
    data_gen = partial(loader.training_batch_generator_full_shot_partial_reset,shot_list=shot_list_train)()


    loader.set_inference_mode(False)

    train_model = build_torch_model(conf)
    #print(train_model)
    if torch.cuda.device_count() > 1:
       print('Using multiple GPUs................',torch.cuda.device_count())
       print('Using multiple GPUs................',torch.cuda.device_count())
       print('Using multiple GPUs................',torch.cuda.device_count())
       print('Using multiple GPUs................',torch.cuda.device_count())
       train_model = nn.DataParallel(train_model)
    else:
       print('Using single GPU..........................................')
    train_model.to(device)
   # try:
      #summary(train_model,(500,14))
   # except:
   #   print('MODEL SUMMARY WARNING!!!!!!!!!!!!!!!!!NOT PASSED for some reason.....')

    num_epochs = conf['training']['num_epochs']
    patience = conf['callbacks']['patience']
    lr_decay = conf['model']['lr_decay']
    lr_decay_factor = conf['model']['lr_decay_factor']
    lr_decay_patience = conf['model']['lr_decay_patience']
    batch_size = 1
    lr = conf['model']['lr']
    clipnorm = conf['model']['clipnorm']
    e = 0


    
    if conf['callbacks']['mode'] == 'max':
        best_so_far = -np.inf
        cmp_fn = max
    else:
        best_so_far = np.inf
        cmp_fn = min
    optimizer = opt.Adam(train_model.parameters(),lr = lr)
    scheduler = opt.lr_scheduler.ExponentialLR(optimizer,lr_decay)
    train_model.train()
    not_updated = 0
    total_loss = 0
    count = 0
    loss_fn = nn.MSELoss(size_average=True)
    model_path = get_model_path(conf)
    makedirs_process_safe(os.path.dirname(model_path))
    epochlog=open('epoch_train_log.txt','w')
    epochlog.write('e,         Train Loss,          Val Loss,          Val ROC\n')
    epochlog.close()
    while e < num_epochs-1:
        print('{} epochs left to go'.format(num_epochs - 1 - e))
        print('\nTraining Epoch {}/{}'.format(e,num_epochs),'starting at',datetime.datetime.now())
        train_model.train()
        scheduler.step()
        (step,curr_loss,ave_loss,num_so_far,effective_epochs) = train_epoch(train_model,data_gen,optimizer,loss_fn,device=device,conf  = conf)
        e = effective_epochs
    
        print('\nFiniehsed Training'.format(e,num_epochs),'finishing at',datetime.datetime.now())
        loader.verbose=False #True during the first iteration
        print('printing_out epoch ', e,'learning rate:',lr)
        for param_group in optimizer.param_groups:
             print(param_group['lr'])

        _,_,_,roc_area,loss = make_predictions_and_evaluate_gpu(conf,shot_list_validate,loader,inference_model=train_model,device=device)
        best_so_far = cmp_fn(roc_area,best_so_far)

        stop_training = False
        print('=========Summary======== for epoch{}'.format(step))
        print('Training Loss numpy: {:.3e}'.format(ave_loss))
        print('Validation Loss: {:.3e}'.format(loss))
        print('Validation ROC: {:.4f}'.format(roc_area))
        epochlog=open('epoch_train_log.txt','a')
        epochlog.write(str(e)+'  '+str(ave_loss)+'   ' +str(loss)+'  '+str(roc_area) +'\n')
        epochlog.close()
        if best_so_far != roc_area: #only save model weights if quantity we are tracking is improving
            print("No improvement, still saving model")
            not_updated += 1
            
            if e > 10 and not_updated>=lr_decay_patience:
                lr /=lr_decay_factor
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
        else:
            print("Saving model")
            not_update=0
            # specific_builder.delete_model_weights(train_model,int(round(e)))
################Saving torch model################################
            torch.save(train_model.state_dict(),model_path[:-3]+'dict.pt')
            torch.save(train_model,model_path)
##################################################################
        if not_updated > patience:
            print("Stopping training due to early stopping")
            break

