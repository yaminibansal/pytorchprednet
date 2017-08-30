import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from prednet.stochastic.train import train
#from prednet.stochastic.evaluation import predict
from prednet.stochastic.models import StochFCDecoder
from prednet.stochastic.losses.MMD import MMDLoss
from prednet.utils.plotting import plot_samples

import time
import math
from prednet.utils.misc import timeSince
import hickle as hkl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

datapath = '/home/ybansal/Documents/Research/pytorchprednet/Data/confused_ball/train.hkl'

if __name__ == "__main__":
    f = open(datapath, 'r')
    data_container = hkl.load(f)
    f.close()
    X_train = np.swapaxes(np.swapaxes(data_container['videos'][:,2], 2, 3), 1, 2)
    Y_train = np.swapaxes(np.swapaxes(data_container['videos'][:,2], 2, 3), 1, 2)
    X_train = Variable(torch.from_numpy(X_train.astype(np.dtype('float32'))), requires_grad=False)
    Y_train = Variable(torch.from_numpy(Y_train.astype(np.dtype('float32'))), requires_grad=False)
    if torch.cuda.is_available():
        X_train = X_train.cuda()
        Y_train = Y_train.cuda()

    num_datapoints = X_train.size()[0]
    num_epochs = 10 #Number of times it goes over the entire training set
    index_array = np.arange(num_datapoints)
    batch_size = 10
    num_batches = num_datapoints/batch_size
    num_samples = 25
    noise_dim = 20

    print_every = 1
    total_loss = 0 # Reset every plot_every iters

    dec_layer_size = [128, 256, 1*20*20]
    model = StochFCDecoder(dec_layer_size)
    if torch.cuda.is_available():
        model.cuda()

    criterion = MMDLoss(1.0)
    optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.9)

    start = time.time()

    for n in range(num_epochs):
        npr.shuffle(index_array)
        
        for b in range(num_batches):
            X_train_batch = Variable(torch.zeros(batch_size, dec_layer_size[0]-noise_dim)).cuda()
            Y_train_batch = Y_train[b*batch_size:(b+1)*batch_size]

            noise = Variable(torch.randn(batch_size, num_samples, noise_dim)).cuda()
            output, loss = train(model, optimizer, criterion, X_train_batch, noise, Y_train_batch)
            total_loss += loss
            
            if b % print_every == 0:
                print('%s (%d %d%%) %.4f' % (timeSince(start), b, b / num_batches * 100, loss))

    plt = plot_samples(Y_train_batch[:,0].data.cpu(), output[:,:,0].data.cpu(), 1, 10)
    plt.show()
