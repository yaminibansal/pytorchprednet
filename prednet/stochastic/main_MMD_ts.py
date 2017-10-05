import argparse

import numpy as np
import numpy.random as npr

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from prednet.stochastic.ts_train import train
from prednet.stochastic.ts_predict import predict
from prednet.stochastic.ts_models import StochLSTMFCEncDec
from prednet.stochastic.losses.MMD import minDistLoss, MMDLoss
from prednet.utils.plotting import plot_stoch_seq
from prednet.utils.misc import timeSince

import time
import math
import hickle as hkl

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

##########################################################
###### Defining the input arguments for the parser #######
##########################################################

parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
parser.add_argument('--dataset', required=True, help='ball')
parser.add_argument('--train_root', required=True, help='path to training data')
parser.add_argument('--val_root', required=True, help='path to validation data')
#parser.add_argument('--test_root', required=True, help='path to test data')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=10, help='Batch Size')
parser.add_argument('--num_samples', type=int, default=50, help='Number of samples produced by generator per datapoint per frame')
parser.add_argument('--modelname', required=True, help='StochLSTMFCEncDec')
parser.add_argument('--hid_size', type=int, required=True)
parser.add_argument('--num_noise_dim', type=int, required=True)
parser.add_argument('--enc_int_layers', type=int, nargs='*', required=True)
parser.add_argument('--dec_int_layers', type=int, nargs='*', required=True)
parser.add_argument('--sig_rbf', type=float, default=1.0, help='RBF kernel bandwidth')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for optimizer Adam')
parser.add_argument('--savepath', help='Path for saving the plots')
parser.add_argument('--num_inp_plts', type=int, help='Number of input data sequences to be plotted')
parser.add_argument('--num_gen_plts', type=int, help='Number of output sample sequences to be plotted per input')
#parser.add_argument('--plot_fn', default=None, help='Name of plotting function to be used, default=None')
#parser.add_argument('--plotargs', default=None, help='Input arguments for plotting function as string in format arg1=val1,arg2=val2...')


opt = parser.parse_args()
print(opt)


#    os.makedirs(opt.outf)
#except OSError:
#    pass

##########################################################
######### Parse arguments and define variables ###########
##########################################################

if opt.dataset == 'ball':
    f = open(opt.train_root, 'r')
    data_container = hkl.load(f)
    f.close()
    X_train = np.swapaxes(np.swapaxes(data_container['videos'][:,0:9], 3, 4), 2, 3)
    Y_train = np.swapaxes(np.swapaxes(data_container['videos'][:,1:10], 3, 4), 2, 3)
    im_dims = X_train.shape[3]*X_train.shape[4]
    X_train = Variable(torch.from_numpy(X_train.astype(np.dtype('float32'))), requires_grad=False)
    Y_train = Variable(torch.from_numpy(Y_train.astype(np.dtype('float32'))), requires_grad=False)
    if torch.cuda.is_available():
        X_train = X_train.cuda()
        Y_train = Y_train.cuda()

    f = open(opt.val_root, 'r')
    data_container = hkl.load(f)
    f.close()
    X_val = np.swapaxes(np.swapaxes(data_container['videos'][:,0:9], 3, 4), 2, 3)
    Y_val = np.swapaxes(np.swapaxes(data_container['videos'][:,1:10], 3, 4), 2, 3)
    X_val = Variable(torch.from_numpy(X_val.astype(np.dtype('float32'))), requires_grad=False)
    Y_val = Variable(torch.from_numpy(Y_val.astype(np.dtype('float32'))), requires_grad=False)
    if torch.cuda.is_available():
        X_val = X_val.cuda()
        Y_val = Y_val.cuda()

    # f = open(opt.test_root, 'r')
    # data_container = hkl.load(f)
    # f.close()
    # X_test = np.swapaxes(np.swapaxes(data_container['videos'][:,0:9], 3, 4), 2, 3)
    # Y_test = np.swapaxes(np.swapaxes(data_container['videos'][:,1:10], 3, 4), 2, 3)
    # X_test = Variable(torch.from_numpy(X_test.astype(np.dtype('float32'))), requires_grad=False)
    # Y_test = Variable(torch.from_numpy(Y_test.astype(np.dtype('float32'))), requires_grad=False)
    # if torch.cuda.is_available():
    #     X_test = X_test.cuda()
    #     Y_test = Y_test.cuda()

    
else:
    raise NotImplementedError

if opt.modelname == 'StochLSTMFCEncDec':
    enc_stack_sizes = np.concatenate(([im_dims], opt.enc_int_layers, [opt.hid_size]))
    #enc_stack_sizes = [1*20*20, 1000, 2000, 40]
    dec_stack_sizes = np.concatenate(([opt.hid_size+opt.num_noise_dim], opt.dec_int_layers, [im_dims]))
    #dec_stack_sizes = [80, 2000, 1000, 1*20*20]
    model = StochLSTMFCEncDec(enc_stack_sizes, dec_stack_sizes, opt.hid_size)

print(model)
num_datapoints = X_train.size()[0]
num_valpoints = X_val.size(0)
num_timesteps = X_train.size()[1]
index_array = np.arange(num_datapoints)
num_batches = num_datapoints/opt.batch_size
print_every = num_batches
total_loss = [] # Reset every plot_every iters
total_val_loss = []

if torch.cuda.is_available():
    model.cuda()

criterion = MMDLoss(opt.sig_rbf)
optimizer = optim.Adam(model.parameters(), lr=opt.lr)

##########################################################
######################## Training ########################
##########################################################

start = time.time()

for n in range(opt.num_epochs):
    npr.shuffle(index_array)
        
    for b in range(num_batches):
        input = X_train[b*opt.batch_size:(b+1)*opt.batch_size]
        target = Y_train[b*opt.batch_size:(b+1)*opt.batch_size]
        noise = Variable(torch.randn(opt.batch_size, opt.num_samples, num_timesteps, opt.num_noise_dim)).cuda()
        output, loss = train(model, optimizer, criterion, input, noise, target)
        total_loss += [loss]
            
        if b % print_every == 0:
            print('Epoch %d: %s (%d %d%%) %.4f' % (n, timeSince(start), b, b / num_batches * 100, loss))


    noise = Variable(torch.randn(num_valpoints, opt.num_samples, num_timesteps, opt.num_noise_dim)).cuda()
    output, hidden, val_loss = predict(model, criterion, X_val, noise, Y_val)
    total_val_loss += [val_loss]


##########################################################
######################Store & Plot########################
##########################################################

plt = plot_stoch_seq(X_val[:,:,0].data.cpu(), output[:,:,:,0].data.cpu(), opt.num_inp_plts, opt.num_gen_plts, seq_ind=[11, 0], savepath=opt.savepath+'samples.png')

torch.save(model, opt.savepath+'model.pt')

output_dict = {}
output_dict['training loss'] = total_loss
output_dict['validation loss'] = total_val_loss
output_dict['hidden states'] = hidden.data.cpu().numpy()

f = open(opt.savepath+'outdata.hkl', 'w')
hkl.dump(output_dict, f)
f.close()


