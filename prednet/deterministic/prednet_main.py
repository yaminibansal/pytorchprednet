import argparse

import numpy as np
import numpy.random as npr

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from prednet.deterministic.ts_models import PredNet
from prednet.deterministic.ts_train import prednet_train 
from prednet.deterministic.ts_predict import prednet_predict
from prednet.utils.plotting import plot_det_seq
from prednet.utils.misc import timeSince
from prednet.utils.data_utils import kittidata

import time
import math
import hickle as hkl

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

##########################################################
###### Defining the input arguments for the parser #######
##########################################################

parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
parser.add_argument('--dataset', required=True, help='kitti')
parser.add_argument('--train_root', required=True, help='path to training data')
parser.add_argument('--train_src_root', required=True, help='path to training data')
parser.add_argument('--val_root', required=True, help='path to validation data')
parser.add_argument('--val_src_root', required=True, help='path to training data')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--samples_per_epoch', type=int, default=100, help='Number of samples trained on per epoch')
parser.add_argument('--batch_size', type=int, default=10, help='Batch Size')
parser.add_argument('--num_t', type=int, required=True, help='number of timesteps')
parser.add_argument('--modelname', required=True, help='PredNet')
parser.add_argument('--enc_filt_size', type=int, nargs='*')
parser.add_argument('--enc_ker_size', type=int, nargs='*')
parser.add_argument('--enc_pool_size', type=int, nargs='*')
parser.add_argument('--hid_filt_size', type=int, nargs='*')
parser.add_argument('--hid_ker_size', type=int, nargs='*')
parser.add_argument('--dec_ker_size', type=int, nargs='*')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for optimizer Adam')
parser.add_argument('--savepath', help='Path for saving the plots')
parser.add_argument('--num_inp_plts', type=int, help='Number of input data sequences to be plotted')

opt = parser.parse_args()
print(opt)

##########################################################
######### Parse arguments and define variables ###########
##########################################################
num_timesteps = opt.num_t

if opt.dataset == 'kitti':
    train_dataset = kittidata(opt.train_root, opt.train_src_root, num_timesteps, opt.samples_per_epoch)
    val_dataset = kittidata(opt.val_root, opt.val_src_root, num_timesteps, opt.samples_per_epoch)
    val_dataloader = DataLoader(val_dataset, 5)
else:
    raise NotImplementedError

if opt.modelname == 'PredNet':
    model = PredNet(opt.enc_filt_size, opt.enc_ker_size, opt.enc_pool_size, opt.hid_filt_size, opt.hid_ker_size, opt.dec_ker_size)

print(model)
num_batches = len(train_dataset)/opt.batch_size
print_every = num_batches
total_loss = [] # Reset every plot_every iters
total_val_loss = []

if torch.cuda.is_available():
    model.cuda()

criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=opt.lr)

start = time.time()
for n in range(opt.num_epochs):
    train_dataset.shuffle() #shuffle data for every epoch and then train sequentially
    train_dataloader = DataLoader(train_dataset, opt.batch_size, shuffle=True)
    
    for b, input in enumerate(train_dataloader):
        input = Variable(input, requires_grad=False)
        target = input
        if torch.cuda.is_available:
            input = input.cuda()
            target = target.cuda()


        output, loss = prednet_train(model, optimizer, criterion, input, target)
        total_loss += [loss]

        if b % print_every == 0:
            print('Epoch %d: %s (%d %d%%) %.4f' % (n, timeSince(start), b, b / num_batches * 100, loss))

# load data for validation loss
i, val_input = enumerate(val_dataloader).next()
val_input = Variable(val_input, requires_grad=False)
if torch.cuda.is_available():
    val_input = val_input.cuda()
val_target = val_input

val_output, val_loss = prednet_predict(model, criterion, val_input, val_target)
            
##########################################################
######################Store & Plot########################
##########################################################

plt = plot_det_seq(val_input[:,:,0].data.cpu(), val_output[:,:,0].data.cpu(), opt.num_inp_plts, seq_ind=[2, 0], savepath=opt.savepath+'samples.png')

torch.save(model, opt.savepath+'model.pt')

output_dict = {}
output_dict['output'] = val_output.data.cpu().numpy()
output_dict['training loss'] = total_loss
output_dict['validation loss'] = val_loss

f = open(opt.savepath+'outdata.hkl', 'w')
hkl.dump(output_dict, f)
f.close()
