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
from prednet.stochastic.discriminators import cndtn_dcgan_dist, cndtn_dcgan_lstm
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
parser.add_argument('--discname', help='cndtn_dcgan_dist | cndtn_dcgan_lstm', required=True)
parser.add_argument('--disc_hid_size', type=int)
parser.add_argument('--disc_enc_size', type=int, nargs='*')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for optimizer Adam')
parser.add_argument('--beta1', type=float, default=0.5, help='Learning rate for optimizer Adam')
parser.add_argument('--savepath', help='Path for saving the plots')
parser.add_argument('--num_inp_plts', type=int, help='Number of input data sequences to be plotted')
parser.add_argument('--num_gen_plts', type=int, help='Number of output sample sequences to be plotted per input')

opt = parser.parse_args()
print(opt)

#try:
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
    im_channels = X_train.shape[2]
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
else:
    raise NotImplementedError

if opt.modelname == 'StochLSTMFCEncDec':
    enc_stack_sizes = np.concatenate(([im_dims], opt.enc_int_layers, [opt.hid_size]))
    dec_stack_sizes = np.concatenate(([opt.hid_size+opt.num_noise_dim], opt.dec_int_layers, [im_dims]))
    gen = StochLSTMFCEncDec(enc_stack_sizes, dec_stack_sizes, opt.hid_size)

print('Generator: ', gen)

if opt.discname == 'cndtn_dcgan_dist':
    num_inp_channels = opt.hid_size + im_channels    
    disc = cndtn_dcgan_dist(num_inp_channels)
elif opt.discname == 'cndtn_dcgan_lstm':
    enc_stack_sizes = np.concatenate(([im_dims], opt.disc_enc_size, [opt.disc_hid_size]))
    disc = cndtn_dcgan_lstm(opt.disc_hid_size, enc_stack_sizes)
else:
    raise NotImplementedError

print('Discriminator: ', disc)

num_datapoints = X_train.size()[0]
num_valpoints = X_val.size(0)
num_timesteps = X_train.size()[1]
index_array = np.arange(num_datapoints)
num_batches = num_datapoints/opt.batch_size
print_every = num_batches
total_gen_loss = []
total_disc_loss = []
total_val_loss = []

criterion = nn.BCELoss()

label = torch.FloatTensor(num_timesteps, opt.batch_size)
label_samples = torch.FloatTensor(num_timesteps, opt.batch_size*opt.num_samples)
real_label = 1
fake_label = 0
    
if torch.cuda.is_available():
    gen.cuda()
    disc.cuda()
    criterion.cuda()
    label = label.cuda()
    label_samples = label_samples.cuda()



optimizerD = optim.Adam(disc.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(gen.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))    

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

        #### Calculate the hidden state for all timesteps
        disc.zero_grad()
        hidden = gen.init_hidden(opt.batch_size)
        hidden_all = Variable(torch.zeros((num_timesteps,)+hidden[0].size()))
        gen_samples_all = Variable(torch.zeros((opt.batch_size*opt.num_samples, num_timesteps)+target[:,0].size()[1:]))
        
        if torch.cuda.is_available:
            hidden_all = hidden_all.cuda()
            gen_samples_all = gen_samples_all.cuda()
            
        errD_real = 0
        errD_fake = 0
        D_x = 0
        D_G_z = 0
        
        for t in range(num_timesteps):
            hidden, gen_output = gen(input[:,t], hidden, noise[:,:,t], target[:,t].size())
            gen_samples = gen_output.contiguous().view((opt.batch_size*opt.num_samples,) + gen_output.size()[2:] )            
            hidden_all[t] = hidden[0]
            gen_samples_all[:,t] = gen_samples

        # Update D
        if opt.discname == 'cndtn_dcgan_dist':
            disc_out_real = disc(target, hidden_all.detach())
        elif opt.discname == 'cndtn_dcgan_lstm':
            disc_out_real = disc(input, target)
        labelv_real = Variable(label.fill_(real_label))
        errD_real = criterion(disc_out_real, labelv_real) #might have to resize
        D_x = disc_out_real.data.mean()
        errD_real.backward()

        if opt.discname == 'cndtn_dcgan_dist':
            disc_out_fake = disc(gen_samples_all.detach(), hidden_all.detach().repeat(1, opt.num_samples, 1)) #not general enough
        elif opt.discname == 'cndtn_dcgan_lstm':
            disc_out_fake = disc(input.repeat(opt.num_samples, 1, 1, 1, 1), gen_samples_all.detach())
        labelv_fake = Variable(label_samples.fill_(fake_label))
        errD_fake += criterion(disc_out_fake, labelv_fake)
        errD_fake.backward()
        D_G_z1 = disc_out_fake.data.mean()
        
        errD = errD_real + errD_fake
        optimizerD.step()

        #Update G
        errG = 0
        gen.zero_grad()
        labelv_real = Variable(label_samples.fill_(real_label))
        if opt.discname == 'cndtn_dcgan_dist':
            disc_out_fake = disc(gen_samples_all, hidden_all.repeat(1, opt.num_samples, 1))
        elif opt.discname == 'cndtn_dcgan_lstm':
            disc_out_fake = disc(input.repeat(opt.num_samples, 1, 1, 1, 1), gen_samples_all)
        errG = criterion(disc_out_fake, labelv_real)
            
        errG.backward()
        D_G_z2 = disc_out_fake.data.mean()
        optimizerG.step()

        print('Epoch: %d, Time: %s (%d %d) Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' %(n, timeSince(start), b, b/num_batches*100, errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))

    noise = Variable(torch.randn(num_valpoints, opt.num_samples, num_timesteps, opt.num_noise_dim)).cuda()
    output, hidden, val_loss = predict(gen, None, X_val, noise, Y_val)
    #total_val_loss += [val_loss]

plt = plot_stoch_seq(X_val[:,:,0].data.cpu(), output[:,:,:,0].data.cpu(), opt.num_inp_plts, opt.num_gen_plts, seq_ind=[11, 0], savepath=opt.savepath+'samples.png')


