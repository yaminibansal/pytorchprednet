import argparse
import os
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

import time
import math
from prednet.utils.misc import timeSince
import hickle as hkl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
parser.add_argument('--dataset', required=True, help='ball')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=10, help='Batch Size')
parser.add_argument('--num_samples', type=int, default=50, help='Number of samples produced by generator per datapoint per frame')
parser.add_argument('--modelname', required=True, help='StochLSTMFCEncDec')
parser.add_argument('--num_noise_dim', type=int, required=True, help='Number of dimensions for Gaussian input noise')
parser.add_argument('--rbf', type=float, default=1.0, help='RBF kernel bandwidth')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for optimizer Adam')
parser.add_argument('--savepath', help='Path for saving the plots')
parser.add_argument('--num_inp_plts', type=int, help='Number of input samples to be plotted')
parser.add_argument('--num_gen_plts', type=int, help='Number of generated samples to be plotted')

opt = parser.parse_args()
print(opt)

#try:
#    os.makedirs(opt.outf)
#except OSError:
#    pass

if opt.dataset == 'ball':
    f = open(opt.dataroot, 'r')
    data_container = hkl.load(f)
    f.close()
    X_train = np.swapaxes(np.swapaxes(data_container['videos'][:,0:9], 3, 4), 2, 3)
    Y_train = np.swapaxes(np.swapaxes(data_container['videos'][:,1:10], 3, 4), 2, 3)
    X_train = Variable(torch.from_numpy(X_train.astype(np.dtype('float32'))), requires_grad=False)
    Y_train = Variable(torch.from_numpy(Y_train.astype(np.dtype('float32'))), requires_grad=False)
    if torch.cuda.is_available():
        X_train = X_train.cuda()
        Y_train = Y_train.cuda()
else:
    raise NotImplementedError

if opt.modelname == 'StochLSTMFCEncDec':
    enc_stack_sizes = [1*20*20, 1000, 2000, 20]
    hid_size = 20
    dec_stack_sizes = [40, 2000, 1000, 1*20*20]
    model = StochLSTMFCEncDec(enc_stack_sizes, dec_stack_sizes, hid_size)

if __name__ == "__main__":

    num_datapoints = X_train.size()[0]
    num_timesteps = X_train.size()[1]
    index_array = np.arange(num_datapoints)
    num_batches = num_datapoints/opt.batch_size
    print_every = 1
    total_loss = 0 # Reset every plot_every iters

    if torch.cuda.is_available():
        model.cuda()

    criterion = MMDLoss(opt.rbf)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    start = time.time()

    for n in range(opt.num_epochs):
        npr.shuffle(index_array)
        
        for b in range(num_batches):
            input = X_train[b*opt.batch_size:(b+1)*opt.batch_size]
            target = Y_train[b*opt.batch_size:(b+1)*opt.batch_size]
            noise = Variable(torch.randn(opt.batch_size, opt.num_samples, num_timesteps, opt.num_noise_dim)).cuda()
            output, loss = train(model, optimizer, criterion, input, noise, target)
            total_loss += loss
            
            if b % print_every == 0:
                print('Epoch %d: %s (%d %d%%) %.4f' % (n, timeSince(start), b, b / num_batches * 100, loss))

    noise = Variable(torch.randn(opt.batch_size, opt.num_samples, num_timesteps, opt.num_noise_dim)).cuda()
    output, loss = predict(model, criterion, X_train[:opt.batch_size], noise, Y_train[:opt.batch_size])

    plt = plot_stoch_seq(X_train[:opt.batch_size,:,0].data.cpu(), output[:,:,:,0].data.cpu(), opt.num_inp_plts, opt.num_gen_plts, savepath=opt.savepath)
    plt.show()
