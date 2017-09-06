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


def plot_samples(samples, num_plts, samp_ind=None, savepath=None):
    '''
    Plots the samples in a suitable grid
    '''
    if len(samples.size()) == 4:
        M = samples.size(0)
        num_samples = samples.size(1)
        samples = samples.contiguous().view(M*num_samples, samples.size(2), samples.size(3))

    print(samples.size())
    
    num_samples = samples.size(0)
    assert num_plts < num_samples

    if samp_ind is None:
        samp_ind = npr.choice(num_samples, num_plts)

    print(samp_ind)
    gs = gridspec.GridSpec(int(math.ceil(math.sqrt(num_plts))), int(math.ceil(math.sqrt(num_plts))))

    for i in range(num_plts):
        plt.subplot(gs[i])
        plt.imshow(samples[i].numpy(), interpolation='none')
        plt.gray()
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')

    if savepath is not None: plt.savefig(savepath)

    return plt

def plot_means(means, im_size, savepath=None):
    num_comps = means.size(0)
    means = means.view(num_comps, im_size[0], im_size[1])

    gs = gridspec.GridSpec(num_comps, 1)

    for i in range(num_comps):
        plt.subplot(gs[i])
        plt.imshow(means[i].numpy(), interpolation='none')
        plt.gray()
        plt.colorbar()
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')

    if savepath is not None: plt.savefig(savepath)

    return plt
        
                   
    

def plot_cond_samples(target, samples, num_inps, num_per_inps, inp_ind=None, samp_ind=None, savepath=None):
    '''
    Plots the ground truth and learnt samples conditioned on that ground truth in some way
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

    print(samp_ind)
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
    N = target.size(0)
    M = output.size(0)
    T = target.size(1)
    assert target.size() == output.size()

    if inp_ind is None:
        ind = npr.choice(N, num_inps)
    else:
        ind = inp_ind
    gs = gridspec.GridSpec(num_inps, T)
    gs.update(wspace=.2, hspace=0.0)
    
    for i in range(num_inps):
        for t in range(T):
            plt.subplot(gs[i*t+t])
            plt.imshow(target[ind[i], t].numpy(), interpolation='none')
            plt.gray()
            plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
            plt.ylabel('Ground Truth', fontsize=10)

            plt.subplot(gs[i+2*t])
            plt.imshow(output[ind[i], t].numpy(), interpolation='none')
            plt.gray()
            plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
            plt.ylabel('Predicted', fontsize=10)
    if savepath is not None: plt.savefig(savepath)
                   
    return plt

def plot_stoch_seq(target, output, num_seqs, seq_ind=None, samp_seq_ind=None):
    '''
    Plots the ground truth sequence and learnt samples
    '''

def plot_loss(plt, loss):
    '''
    Plots the loss function during training
    '''
    
    
