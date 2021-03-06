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
from .models import * #FTCN #FLSTM

#from torch_QConv_Kernel import device

model_filename = 'torch_modelfull_model.pt'

def build_torch_model(conf, args):
    n_scalars, n_profiles, profile_size = get_signal_dimensions(conf)
    channels_spatial = args.channels_spatial
    kernel_spatial = args.kernel_spatial
    linear_sizes = args.linear_sizes
    channels_temporal = args.channels_temporal
    kernel_temporal = args.kernel_temporal
    ancillas = args.ancillas
    output_size = 1
    dropout = conf['model']['dropout_prob']
    if args.no_scalars:
        n_scalars = 0
        print('NO SCALARS ARE USED, ONLY 1D SIGNALS')
    model = FTCN(n_scalars, n_profiles, profile_size, channels_spatial, kernel_spatial, linear_sizes, channels_temporal, kernel_temporal, output_size, ancillas=ancillas, dropout=dropout)
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



def make_predictions(conf, shot_list, loader, args, custom_path=None, inference_model=None, device=None):
    no_scalars = args.no_scalars
    input_div = args.input_div
    subsampling = args.subsampling

    n_scalars, n_profiles, profile_size = get_signal_dimensions(conf)
    generator = loader.inference_batch_generator_full_shot(shot_list)
    if inference_model == None:
      if custom_path == None:
        model_path = get_model_path(conf)
      else:
        model_path = custom_path
      print('model-path is: ',model_path)
      inference_model=torch.load(model_path)
    #shot_list = shot_list.random_sublist(10)
    inference_model.eval()
    y_prime = []
    y_gold = []
    disruptive = []
    num_shots = len(shot_list)

    while True:
        x, y, mask, disr, lengths, num_so_far, num_total = next(generator)
        x, y, mask = x[:,::subsampling], y[:,::subsampling], mask[:,::subsampling]
        if input_div != 1.0:
            x[:,:,n_scalars:] = x[:,:,n_scalars:]/input_div
        if no_scalars:
            x = x[:,:,n_scalars:]
        output = apply_model_to_np(inference_model, x, device=device)
        for batch_idx in range(x.shape[0]):
            curr_length = 1+lengths[batch_idx]//10
            y_prime += [output[batch_idx,:curr_length,:]]
            y_gold += [y[batch_idx,:curr_length,:]]
            disruptive += [disr[batch_idx]]
        if len(disruptive) >= num_shots:
            y_prime = y_prime[:num_shots]
            y_gold = y_gold[:num_shots]
            disruptive = disruptive[:num_shots]
            break
    return y_prime, y_gold, disruptive

def make_predictions_and_evaluate_gpu(conf ,shot_list, loader, args, custom_path=None, inference_model=None, device=None):
    y_prime,y_gold,disruptive = make_predictions(conf, shot_list, loader, args, custom_path=custom_path, inference_model=inference_model, device=device)
    print('y_prime,', len(y_prime), len(y_prime[0]))
    print('y_gold,', len(y_gold), len(y_gold[0]))
    print('disruptive,', len(disruptive))
    analyzer = PerformanceAnalyzer(conf=conf)
    roc_area = analyzer.get_roc_area(y_prime, y_gold, disruptive)
    #roc_area=0.0
    loss = get_loss_from_list(y_prime,y_gold,conf['data']['target'])
    return y_prime,y_gold,disruptive,roc_area,loss


def get_model_path(conf):
    return conf['paths']['model_save_path']  + model_filename #save_prepath + model_filename


def train_epoch(model, data_gen, optimizer, loss_fn, args, device=None, conf={}):
    n_scalars, n_profiles, profile_size = get_signal_dimensions(conf)
    subsampling =  args.subsampling
    input_div = args.input_div
    loss = 0
    total_loss = 0
    num_so_far = 0
    x_,y_,mask_,num_so_far_start,num_total = next(data_gen)
    if 'noise' in conf['training'].keys() and conf['training']['noise']!=False:
            x_=add_noise(x_,conf = conf)
    num_so_far = num_so_far_start
    step = 0

    while True:
        x, y, mask = Variable(torch.from_numpy(x_).float()).to(device), Variable(torch.from_numpy(y_).float()).to(device),Variable(torch.from_numpy(mask_).byte()).to(device).to(torch.bool)
        # Time subsampling is used to reduce memory use and computation time.
        x, y, mask = x.unfold(1, 1, subsampling).squeeze(-1), y.unfold(1, 1, subsampling).squeeze(-1), mask.unfold(1, 1, subsampling).squeeze(-1)
        # Division is used to scale the inputs for the spatial convolution layers. This is becuase for quantum layers, the input range should be a rotation angle. Although arctan can transform the range of the data to between -1 and 1, inputs that are larger than 2 (less than -2) are squeezed to values really close to 1 (-1). Many data points are larger than 2 and this results in a loss of information.
        if input_div != 1.0:
            x[:,:,n_scalars:] = x[:,:,n_scalars]/input_div
        if args.no_scalars:
            x = x[:,:,n_scalars:]
        optimizer.zero_grad()
        output = model(x)
        output_masked = torch.masked_select(output,mask)
        y_masked = torch.masked_select(y,mask)

        loss = loss_fn(output_masked,y_masked)
        if loss < 20:
            total_loss += loss.data.item()        
            loss.backward()
            optimizer.step()
            step += 1
            print("[{}]  [{}/{}] loss: {:.3f}, ave_loss: {:.3f}".format(step,num_so_far-num_so_far_start,num_total,loss.data.item(),total_loss/step))
        else:
            if step == 0:
                total_loss = 0
                step += 1
            else:
                total_loss = total_loss/step
                step += 1
                total_loss = total_loss*step
            print(loss)
            print("[{}]  [{}/{}] loss anomaly, ave_loss: {:.3f}".format(step,num_so_far-num_so_far_start,num_total,total_loss/step))
        
        if num_so_far-num_so_far_start >= num_total:
            break
        x_,y_,mask_,num_so_far,num_total = next(data_gen)
    return step,loss.data.item(),total_loss/step,num_so_far,1.0*num_so_far/num_total


def train(conf, shot_list_train, shot_list_validate, loader, args):
    # Initialize model
    np.random.seed(1)
    use_cuda=True
    if torch.cuda.is_available() and use_cuda:
        device = torch.device('cuda:0')
    else:
        device = 'cpu'

    data_gen = partial(loader.training_batch_generator_full_shot_partial_reset,shot_list=shot_list_train)()
    loader.set_inference_mode(False)

    train_model = build_torch_model(conf, args)
    if torch.cuda.device_count() > 1:
       train_model = nn.DataParallel(train_model)
    train_model.to(device)

    # Initialize optimizer
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

    # Training
    train_model.train()
    not_updated = 0
    total_loss = 0
    count = 0
    loss_fn = nn.MSELoss()
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
        (step,curr_loss,ave_loss,num_so_far,effective_epochs) = train_epoch(train_model, data_gen, optimizer, loss_fn, args, device=device, conf=conf)
        e = effective_epochs
    
        print('\nFinished Training'.format(e,num_epochs),'finishing at',datetime.datetime.now())
        loader.verbose=False #True during the first iteration
        print('printing_out epoch ', e,'learning rate:',lr)
        for param_group in optimizer.param_groups:
             print(param_group['lr'])

        _,_,_,roc_area,loss = make_predictions_and_evaluate_gpu(conf, shot_list_validate, loader, args, inference_model=train_model, device=device)
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
################Saving torch model################################
            torch.save(train_model.state_dict(),model_path[:-3]+'dict.pt')
            torch.save(train_model,model_path)
##################################################################
        if not_updated > patience:
            print("Stopping training due to early stopping")
            break