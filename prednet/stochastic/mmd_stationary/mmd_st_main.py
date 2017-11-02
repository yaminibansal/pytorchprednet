import argparse
import os
import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from prednet.stochastic.st_train import train
from prednet.stochastic.models.models import StochFCDecoder, MeanFinder, GMM
from prednet.stochastic.losses.MMD import minDistLoss, MMDLoss
from prednet.utils.plotting import plot_samples, plot_means, plot_2dviz
from prednet.utils.logger import Logger

import time
import math
from prednet.utils.misc import timeSince
import hickle as hkl
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

##########################################################
###### Defining the input arguments for the parser #######
##########################################################

parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
parser.add_argument('--randomseed', required=True, help='Give -1 to pick seed randomly')
# Training data
parser.add_argument('--dataset', required=True, help='ball | distribution | video')
parser.add_argument('--train_root', required=True, help='path to training data')
parser.add_argument('--val_root', required=True, help='path to validation data')
# Training arguments
parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=10, help='Batch Size')
parser.add_argument('--num_samples', type=int, default=50, help='Number of samples produced by generator per datapoint')
# Models
parser.add_argument('--modelname', required=True, help='StochFCDecoder | MeanFinder')
parser.add_argument('--hid_size', type=int, required=True)
parser.add_argument('--num_noise_dim', type=int, required=True)
parser.add_argument('--rbf', type=float, default=1.0, help='RBF kernel bandwidth')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for optimizer Adam')
# Plotting and saving
parser.add_argument('--log', action='store_true')
parser.add_argument('--logsteps', type=int, help='Log every logsteps for viz')
parser.add_argument('--showplot', action='store_true', help='Show final plots')
parser.add_argument('--saveplot', action='store_true', help='Save final plots')
parser.add_argument('--savedata', action='store_true', help='Save final model etc')
parser.add_argument('--savepath', help='Path to directory for saving output data and plots')
parser.add_argument('--num_inp_plts', type=int, help='Number of input samples to be plotted')
parser.add_argument('--num_gen_plts', type=int, help='Number of generated samples to be plotted')

opt = parser.parse_args()
print(opt)

def to_np(x):
    return x.data.cpu().numpy()

##########################################################
######### Parse arguments and define variables ###########
##########################################################

if opt.randomseed=='-1':
    rseed = npr.randint(5000)
else:
    rseed = opt.randomseed
npr.seed(rseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(rseed)
else:
    torch.manual_seed(rseed)

# Making dir to save things if required
if opt.saveplot or opt.savedata or opt.log:
    try:
        os.makedirs(opt.savepath)
        os.makedirs(opt.savepath+'/logs')
        os.makedirs(opt.savepath+'/2dviz')
    except OSError:
        pass

##### Dataset
###### Dataset
if opt.dataset == 'ball':
    f = open(opt.train_root, 'r')
    data_container = hkl.load(f)
    f.close()
    X_train = np.swapaxes(np.swapaxes(data_container['videos'][:,0], 2, 3), 1, 2)
    X_train = Variable(torch.from_numpy(X_train.astype(np.dtype('float32'))), requires_grad=False)
    if torch.cuda.is_available():
        X_train = X_train.cuda()

    f = open(opt.val_root, 'r')
    data_container = hkl.load(f)
    f.close()
    X_val = np.swapaxes(np.swapaxes(data_container['videos'][:,0], 2, 3), 1, 2)
    X_val = Variable(torch.from_numpy(X_val.astype(np.dtype('float32'))), requires_grad=False)
    if torch.cuda.is_available():
        X_val = X_val.cuda()

# elif opt.dataset == 'ball2pos':
#     # Validation not implemented
#     f = open(opt.dataroot, 'r')
#     data_container = hkl.load(f)
#     f.close()
#     X_train = data_container['images']
#     Y_train = data_container['images']
#     X_train = Variable(torch.from_numpy(X_train.astype(np.dtype('float32'))), requires_grad=False)
#     Y_train = Variable(torch.from_numpy(Y_train.astype(np.dtype('float32'))), requires_grad=False)
#     if torch.cuda.is_available():
#         X_train = X_train.cuda()
#         Y_train = Y_train.cuda()    

# elif opt.dataset == 'gaussian':
#     # Validation not implemented
#     f = open(opt.dataroot, 'r')
#     data_container = hkl.load(f)
#     f.close()
#     X_train = data_container['data']
#     Y_train = data_container['data']
#     X_train = Variable(torch.from_numpy(X_train.astype(np.dtype('float32'))), requires_grad=False)
#     Y_train = Variable(torch.from_numpy(Y_train.astype(np.dtype('float32'))), requires_grad=False)
#     if torch.cuda.is_available():
#         X_train = X_train.cuda()
#         Y_train = Y_train.cuda() 
else:
    raise NotImplementedError

if opt.modelname == 'StochFCDecoder':
    dec_layer_size = [opt.hid_size+opt.num_noise_dim, 2000, 1000, 1*20*20]
    gen = StochFCDecoder(dec_layer_size)
    if opt.hid_size>0:
        input = Variable(torch.zeros(opt.batch_size, opt.hid_size)).cuda()
    else:
        input = None
elif opt.modelname == 'MeanFinder':
    gen = MeanFinder(1*20*20)
    input = None
elif opt.modelname == 'GMM':
    gen = GMM(1*20*20, 5)
    input = None
print('Generator: ', gen)

num_datapoints = X_train.size()[0]
num_valpoints = X_val.size()[0]
index_array = np.arange(num_datapoints)
num_batches = num_datapoints/opt.batch_size
noise_dim = 2

print_every = num_batches
total_loss = []
total_val_loss = []
log_every = opt.logsteps
if opt.log: logger = Logger(opt.savepath+'/logs', str(rseed))

criterion = MMDLoss(opt.rbf)
optimizer = optim.Adam(gen.parameters(), lr=opt.lr)

if torch.cuda.is_available():
    gen.cuda()
    criterion.cuda()

##########################################################
######################## Training ########################
##########################################################

start = time.time()
step = -1
for n in range(opt.num_epochs):
    npr.shuffle(index_array)
        
    for b in range(num_batches):
        step += 1
        target = X_train[b*opt.batch_size:(b+1)*opt.batch_size]
        noise = Variable(torch.randn(opt.batch_size, opt.num_samples, opt.num_noise_dim)).cuda()
        output, loss = train(gen, optimizer, criterion, input, noise, target)
        total_loss += [loss]
            
        if b % print_every == 0:
            print('%d: %s (%d %d%%) %.4f' % (n, timeSince(start), b, b / num_batches * 100, loss))
        ##########################################################
        ################ Logging for TF Viz ######################
        ##########################################################
        if opt.log:
            if step%log_every==0:

                #============ TensorBoard logging ============#
                # (1) Log the scalar values
                info = {
                    'loss': loss,
                }

                for tag, value in info.items():
                    logger.scalar_summary(tag, value, step)

                # (2) Log values and gradients of the parameters (histogram)
                for tag, value in gen.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary(tag, to_np(value), step)
                    logger.histo_summary(tag+'/grad', to_np(value.grad), step)

                if opt.hid_size > 0:
                    val_input = Variable(torch.zeros(num_valpoints, opt.hid_size)).cuda()
                else:
                    val_input = None
                val_noise = Variable(torch.randn(num_valpoints, opt.num_samples, opt.num_noise_dim)).cuda()
                val_samples_orig = gen(val_input, val_noise, (num_valpoints,)+target.size()[1:])
                val_samples = val_samples_orig.contiguous().view(num_valpoints*opt.num_samples, target.size(1), target.size(2), target.size(3))
                basis_1 = np.zeros((20, 20))
                basis_1[10:13, 0:3] = 1.
                basis_1 = basis_1.reshape((400, 1))
                basis_2 = np.zeros((20, 20))
                basis_2[10:13, 4:7] = 1.
                basis_2 = basis_2.reshape((400, 1))
                basis = np.concatenate((basis_1, basis_2), axis=1)
                plt = plot_2dviz(val_samples.contiguous().data.cpu().view(num_valpoints*opt.num_samples, -1), basis, showplot=False, savepath=opt.savepath+'/2dviz/'+str(step)+'_2dviz.png')

##########################################################
################ Saving and plotting #####################
##########################################################
if opt.saveplot:
    plt = plot_samples(val_samples_orig[:,:,0].data.cpu(), opt.num_gen_plts, savepath=opt.savepath+'/outputsamples.png')
    plt = plot_2dviz(val_samples.contiguous().data.cpu().view(num_valpoints*opt.num_samples, -1), basis, showplot=False, savepath=opt.savepath+'/2dviz.png')
if opt.showplot:
    plt = plot_samples(val_samples_orig[:,:,0].data.cpu(), opt.num_gen_plts, savepath=None)
    plt.show()

if opt.savedata:
    save_dict = {}
    fn = opt.savepath+'/'+str(rseed)+'outputs.pkl'
    save_dict['params'] = opt
    save_dict['seed'] = rseed
    f = open(fn, 'w')
    pkl.dump(save_dict, f)
    f.close()
    # Not saving model or losses yet
