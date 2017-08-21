import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from prednet.stochastic.models import StochLSTMEncDec
from prednet.stochastic.losses.MMD import MMDLossFn

import time
import math
import hickle as hkl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

datapath = '/home/ybansal/Documents/Research/pytorchprednet/Data/confused_ball/train.hkl'


def train(model, optimizer, criterion, X_train_batch, Y_train_batch, num_samples):

    batch_size = X_train_batch.size()[0]
    im_size = (X_train_batch.size()[3], X_train_batch.size()[4])
    
    hidden = model.init_hidden(batch_size)
    
    optimizer.zero_grad()

    loss = 0

    for i in range(X_train_batch.size()[1]):
        hidden, output = model(X_train_batch[:,i], hidden, num_samples)
        for j in range(X_train_batch.size()[0]):
            loss += criterion.forward( output[j].view(output[j].size()[0], -1), Y_train_batch[j,i].view(1, -1) )

    loss.backward()
    optimizer.step()

    return output, loss.data[0] / X_train_batch.size()[0]

def predict(model, X_train, num_samples):
    N = X_train.size()[0]
    
    predicted_frames = Variable(torch.Tensor(N, num_samples, num_timesteps, 1, im_size[0], im_size[1]))
    predicted_frames = predicted_frames.cuda()
    
    hidden = model.init_hidden(N)
        
    hidden = (hidden[0].cuda(), hidden[1].cuda())
    
    for i in range(X_train_batch.size()[1]):
        hidden, predicted_frames[:,:,i] = model(X_train[:,i], hidden, num_samples)
        
    return predicted_frames

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

if __name__=="__main__":
    f = open(datapath, 'r')
    data_container = hkl.load(f)
    f.close()
    X_train = np.swapaxes(np.swapaxes(data_container['videos'][:,0:9], 3, 4), 2, 3)
    Y_train = np.swapaxes(np.swapaxes(data_container['videos'][:,1:10], 3, 4), 2, 3)
    X_train = Variable(torch.from_numpy(X_train.astype(np.dtype('float32'))), requires_grad=False)
    Y_train = Variable(torch.from_numpy(Y_train.astype(np.dtype('float32'))), requires_grad=False)
    if torch.cuda.is_available():
        X_train = X_train.cuda()
        Y_train = Y_train.cuda()

    num_datapoints = X_train.size()[0]
    num_timesteps = X_train.size()[1]
    im_size = (X_train.size()[3], X_train.size()[4])

    num_epochs = 1 #Number of times it goes over the entire training set
    index_array = np.arange(num_datapoints)
    batch_size = 5
    num_batches = num_datapoints/batch_size


    print_every = 1
    total_loss = 0 # Reset every plot_every iters

    enc_filt_size = (1, 64, 128)
    enc_ker_size = (3, 3, 3)
    enc_pool_size = (2, 2, 2)
    hid_size = (im_size[0]/4)*(im_size[1]/4)*enc_filt_size[2]
    dec_filt_size = (128, 64, 32)
    dec_ker_size = (3, 3, 3)
    dec_upsample_size = (2, 2, 2)
    lstm_inp_size = (im_size[0]/4)*(im_size[1]/4)*enc_filt_size[2]
    num_samples = 500
    num_noise_dims = 20
    
    model = StochLSTMEncDec(enc_filt_size, enc_ker_size, enc_pool_size, hid_size, dec_filt_size, dec_ker_size, dec_upsample_size, lstm_inp_size, num_noise_dims)

    if torch.cuda.is_available():
        model.cuda()
    
    criterion = MMDLossFn(100.0)
    optimizer = optim.RMSprop(model.parameters(), lr=0.0001, alpha=0.8)

    start = time.time()

    for n in range(num_epochs):
        npr.shuffle(index_array)
        
        for b in range(num_batches):
            X_train_batch = X_train[b*batch_size:(b+1)*batch_size]
            Y_train_batch = Y_train[b*batch_size:(b+1)*batch_size]

            output, loss = train(model, optimizer, criterion, X_train_batch, Y_train_batch, num_samples)
            total_loss += loss
            
            if b % print_every == 0:
                print('%s (%d %d%%) %.4f' % (timeSince(start), b, b / num_batches * 100, loss))

    i = 7
    num_pred_samples = 5
    predicted_frames = predict(model, X_train[:20], num_pred_samples)
    nt = 9
    gs = gridspec.GridSpec(3, nt)
    gs.update(wspace=0., hspace=0.)
    for t in range(nt):
        plt.subplot(gs[t])
        plt.imshow(X_train[i,t,0,:,:].data.cpu().numpy(), interpolation='none')
        plt.gray()
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        if t==0: plt.ylabel('Actual', fontsize=10)

        plt.subplot(gs[t + nt])
        plt.imshow(predicted_frames[i,0,t,0,:,:].data.cpu().numpy(), interpolation='none')
        plt.gray()
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        if t==0: plt.ylabel('Predicted', fontsize=10)
            
        plt.subplot(gs[t + 2*nt])
        plt.imshow(predicted_frames[i,1,t,0,:,:].data.cpu().numpy(), interpolation='none')
        plt.gray()
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        if t==0: plt.ylabel('Predicted', fontsize=10)
            
    plt.show()
