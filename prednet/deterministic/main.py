import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from prednet.deterministic.train import train
from prednet.deterministic.evaluation import predict
from prednet.deterministic.models import FCDecoder

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

    print_every = 1
    total_loss = 0 # Reset every plot_every iters

    dec_layer_size = [128, 256, 1*20*20]
    model = FCDecoder(dec_layer_size)
    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9)

    start = time.time()

    for n in range(num_epochs):
        npr.shuffle(index_array)
        
        for b in range(num_batches):
            X_train_batch = Variable(torch.zeros(batch_size, dec_layer_size[0])).cuda()
            Y_train_batch = Y_train[b*batch_size:(b+1)*batch_size]

            output, loss = train(model, optimizer, criterion, X_train_batch, Y_train_batch)
            total_loss += loss
            
            if b % print_every == 0:
                print('%s (%d %d%%) %.4f' % (timeSince(start), b, b / num_batches * 100, loss))

    i = 7
    num_pred_samples = 20
    predicted_frames, loss = predict(model, criterion, Variable(torch.zeros(num_pred_samples, dec_layer_size[0])).cuda(), X_train[:num_pred_samples])
    nt = 9
    gs = gridspec.GridSpec(int(math.ceil(math.sqrt(num_pred_samples))), int(math.ceil(math.sqrt(num_pred_samples))))
    gs.update(wspace=0., hspace=0.)

    for n in range(num_pred_samples):
        plt.subplot(gs[n])
        plt.imshow(predicted_frames[n,0,:,:].data.cpu().numpy(), interpolation='none')
        plt.gray()
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')

    plt.show()
