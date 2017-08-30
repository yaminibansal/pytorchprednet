import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import numpy.random as npr
import math

def plot_output(target, output, num_inps, inp_ind=None, savepath=None):
    '''
    Plots the ground truth and the learnt output

    target: 3 dimensional input with diff images accros dim 0
    inputs are torch tensors
    '''
    N = target.size(0)
    M = output.size(0)
    assert N == M

    if inp_ind is None:
        ind = npr.choice(N, num_inps)
    else:
        ind = inp_ind
    gs = gridspec.GridSpec(num_inps, 2)
    gs.update(wspace=.2, hspace=0.0)
    
    for i in range(num_inps):
        plt.subplot(gs[2*i])
        plt.imshow(target[ind[i]].numpy(), interpolation='none')
        plt.gray()
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        plt.ylabel('Ground Truth', fontsize=10)

        plt.subplot(gs[2*i+1])
        plt.imshow(output[ind[i]].numpy(), interpolation='none')
        plt.gray()
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        plt.ylabel('Predicted', fontsize=10)
    if savepath is not None: plt.savefig(savepath)
                   
    return plt
                   
    

def plot_samples(target, samples, num_inps, num_per_inps, inp_ind=None, samp_ind=None, savepath=None):
    '''
    Plots the ground truth and learnt samples
    '''
    assert len(target.size()) + 1 == len(samples.size())
    N = target.size(0)
    M = samples.size(0)
    num_samples = samples.size(1)
    assert N == M

    if inp_ind is None:
        ind = npr.choice(N, num_inps)
    else:
        ind = inp_ind
        
    if samp_ind is None:
        samp_ind = np.zeros((num_inps, num_per_inps))
        for i in range(num_inps):
            samp_ind[i,:] = np.array(npr.choice(num_samples, num_per_inps))

    gs = gridspec.GridSpec(num_inps, num_per_inps+1)

    for i in range(num_inps):
        plt.subplot(gs[(num_per_inps+1)*i])
        plt.imshow(target[ind[i]].numpy(), interpolation='none')
        plt.gray()
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        plt.ylabel('Ground Truth', fontsize=10)

        for m in range(num_per_inps):
            plt.subplot(gs[(num_per_inps+1)*i+m+1])
            plt.imshow(samples[ind[i], samp_ind[i][m]].numpy(), interpolation='none')
            plt.gray()
            plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')

    if savepath is not None: plt.savefig(savepath)
                   
    return plt


                   
                   


def plot_det_seq(target, output, num_seqs, seq_ind=None):
    '''
    Plots the ground truth sequence and the learnt sequence
    '''

def plot_stoch_seq(target, output, num_seqs, seq_ind=None, samp_seq_ind=None):
    '''
    Plots the ground truth sequence and learnt samples
    '''

def plot_loss(plt, loss):
    '''
    Plots the loss function during training
    '''
    
    
