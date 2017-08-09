import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from prednet.deterministic.models import ConvLSTMEncDec
import time
import math
import hickle as hkl
datapath = '/home/ybansal/Documents/Research/pytorchprednet/Data/confused_ball/train.hkl'


def train(model, optimizer, criterion, X_train_batch, Y_train_batch):

    batch_size = X_train_batch.size()[0]
    im_size = (X_train_batch.size()[2], X_train_batch.size()[3])
    
    R1_hidden = model.convLSTM1.init_hidden(batch_size, (im_size[0]/2, im_size[0]/2))
    R0_hidden = model.convLSTM0.init_hidden(batch_size, im_size)
    R1_hidden = (R1_hidden[0].cuda(), R1_hidden[1].cuda())
    R0_hidden = (R0_hidden[0].cuda(), R0_hidden[1].cuda())
    
    optimizer.zero_grad()

    loss = 0

    for i in range(X_train_batch.size()[1]):
        R1_hidden, R0_hidden, output = model(X_train_batch[:,i], R1_hidden, R0_hidden)
        loss += criterion(output, Y_train_batch[:,i])

    loss.backward()
    optimizer.step()

    return output, loss.data[0] / X_train_batch.size()[0]

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

    num_epochs = 10 #Number of times it goes over the entire training set
    index_array = np.arange(num_datapoints)
    batch_size = 10
    num_batches = num_datapoints/batch_size


    print_every = 1
    total_loss = 0 # Reset every plot_every iters
    
    model = ConvLSTMEncDec()
    if torch.cuda.is_available():
        model.cuda()
    
    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9)

    start = time.time()

    for n in range(num_epochs):
        npr.shuffle(index_array)
        
        for b in range(num_batches):
            X_train_batch = X_train[b*batch_size:(b+1)*batch_size]
            Y_train_batch = Y_train[b*batch_size:(b+1)*batch_size]

            output, loss = train(model, optimizer, criterion, X_train_batch, Y_train_batch)
            total_loss += loss
            
            if b % print_every == 0:
                print('%s (%d %d%%) %.4f' % (timeSince(start), b, b / num_batches * 100, loss))
