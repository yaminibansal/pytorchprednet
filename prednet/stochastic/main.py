import argparse
import os
import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from prednet.stochastic.train import train
#from prednet.stochastic.evaluation import predict
from prednet.stochastic.models import StochFCDecoder, MeanFinder, GMM
from prednet.stochastic.losses.MMD import minDistLoss, MMDLoss
from prednet.utils.plotting import plot_samples, plot_means

import time
import math
from prednet.utils.misc import timeSince
import hickle as hkl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
parser.add_argument('--dataset', required=True, help='ball | ball2pos')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=10, help='Batch Size')
parser.add_argument('--num_samples', type=int, default=50, help='Number of samples produced by generator per datapoint')
parser.add_argument('--modelname', required=True, help='StochFCDecoder | MeanFinder')
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
    X_train = np.swapaxes(np.swapaxes(data_container['videos'][:,2], 2, 3), 1, 2)
    Y_train = np.swapaxes(np.swapaxes(data_container['videos'][:,2], 2, 3), 1, 2)
    X_train = Variable(torch.from_numpy(X_train.astype(np.dtype('float32'))), requires_grad=False)
    Y_train = Variable(torch.from_numpy(Y_train.astype(np.dtype('float32'))), requires_grad=False)
    if torch.cuda.is_available():
        X_train = X_train.cuda()
        Y_train = Y_train.cuda()
        
elif opt.dataset == 'ball2pos':
    f = open(opt.dataroot, 'r')
    data_container = hkl.load(f)
    f.close()
    X_train = data_container['images']
    Y_train = data_container['images']
    X_train = Variable(torch.from_numpy(X_train.astype(np.dtype('float32'))), requires_grad=False)
    Y_train = Variable(torch.from_numpy(Y_train.astype(np.dtype('float32'))), requires_grad=False)
    if torch.cuda.is_available():
        X_train = X_train.cuda()
        Y_train = Y_train.cuda()    
        
else:
    raise NotImplementedError

if opt.modelname == 'StochFCDecoder':
    dec_layer_size = [22, 2000, 1000, 1*20*20]
    model = StochFCDecoder(dec_layer_size)
    input = Variable(torch.zeros(opt.batch_size, dec_layer_size[0]-opt.num_noise_dim)).cuda()
elif opt.modelname == 'MeanFinder':
    model = MeanFinder(1*20*20)
    input = None
elif opt.modelname == 'GMM':
    model = GMM(1*20*20, 5)
    input = None


if __name__ == "__main__":

    num_datapoints = X_train.size()[0]
    index_array = np.arange(num_datapoints)
    num_batches = num_datapoints/opt.batch_size
    noise_dim = 2
    
    # if opt.num_inp_plts > num_datapoints:
    #     raise ValueError
    # if opt.num_gen_plts > opt.num_samples:
    #     raise ValueError

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
            Y_train_batch = Y_train[b*opt.batch_size:(b+1)*opt.batch_size]
            noise = Variable(torch.randn(opt.batch_size, opt.num_samples, opt.num_noise_dim)).cuda() #Variable(torch.zeros(opt.batch_size, opt.num_samples, opt.num_noise_dim)).cuda()    

            output, loss = train(model, optimizer, criterion, input, noise, Y_train_batch)
            total_loss += loss
            
            if b % print_every == 0:
                print('%s (%d %d%%) %.4f' % (timeSince(start), b, b / num_batches * 100, loss))

    #plt = plot_means(model.means.data.cpu(), (output.size(3), output.size(4)), savepath=opt.savepath)
    plt = plot_samples(output[:,:,0].data.cpu(), opt.num_gen_plts, savepath=opt.savepath)
    plt.show()