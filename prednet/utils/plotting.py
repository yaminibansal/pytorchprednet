import numpy as np
import numpy.random as npr
import math

def plot_output(target, output, num_inps, showplot=True, inp_ind=None, savepath=None):
    '''
    Plots the ground truth and the learnt output

    target: 3 dimensional input with diff images accros dim 0
    inputs are torch tensors
    '''
    if not showplot:
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        plt.ioff() 
        import matplotlib.gridspec as gridspec
    else:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        plt.ion()
    plt.clf()

    
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


def plot_samples(samples, num_plts, showplot=True, samp_ind=None, savepath=None):
    '''
    Plots the samples in a suitable grid
    '''
    if not showplot:
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        plt.ioff() 
        import matplotlib.gridspec as gridspec
    else:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        plt.ion()
    plt.clf()
    
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

def plot_means(means, im_size, showplot=True, savepath=None):
    if not showplot:
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        plt.ioff() 
        import matplotlib.gridspec as gridspec
    else:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        plt.ion()
    plt.clf()
        
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
        
                   
    

def plot_cond_samples(target, samples, num_inps, num_per_inps, showplot=True, inp_ind=None, samp_ind=None, savepath=None):
    '''
    Plots the ground truth and learnt samples conditioned on that ground truth in some way
    '''
    if not showplot:
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        plt.ioff() 
        import matplotlib.gridspec as gridspec
    else:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        plt.ion()
    plt.clf()
        
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


                   
                   


def plot_det_seq(target, output, num_seqs, showplot=True, seq_ind=None, savepath=None):
    '''
    Plots the ground truth sequence and the learnt sequence
    '''
    if not showplot:
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        plt.ioff() 
        import matplotlib.gridspec as gridspec
    else:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        plt.ion()
    plt.clf()
        
    N = target.size(0)
    M = output.size(0)
    T = target.size(1)
    print(target.size(), output.size())
    assert target.size() == output.size()

    if seq_ind is None:
        ind = npr.choice(N, num_seqs)
    else:
        ind = seq_ind
    gs = gridspec.GridSpec(num_seqs*2, T)
    gs.update(wspace=.2, hspace=0.0)
    
    for i in range(num_seqs):
        for t in range(T):
            plt.subplot(gs[2*i*T+t])
            plt.imshow(target[ind[i], t].numpy(), interpolation='none')
            plt.gray()
            plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
            if t==0: plt.ylabel('Ground Truth', fontsize=10)

            plt.subplot(gs[2*i*T+T+t])
            plt.imshow(output[ind[i], t].numpy(), interpolation='none')
            plt.gray()
            plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
            if t==0: plt.ylabel('Predicted', fontsize=10)
    if savepath is not None: plt.savefig(savepath)
                   
    return plt

def plot_stoch_seq(target, samples, num_seqs, num_per_seq, showplot=True, seq_ind=None, samp_seq_ind=None, savepath=None):
    '''
    Plots the ground truth sequence and learnt samples
    '''
    if not showplot:
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        plt.ioff() 
        import matplotlib.gridspec as gridspec
    else:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        plt.ion()
    plt.clf()
        
    N = target.size(0)
    M = samples.size(0)
    T = target.size(1)+1
    assert N == M
    num_samples = samples.size(1)

    if seq_ind is None:
        ind = npr.choice(N, num_seqs)
    else:
        ind = seq_ind

    if samp_seq_ind is None:
        samp_seq_ind = np.zeros((num_seqs, num_per_seq))
        for i in range(num_seqs):
            samp_seq_ind[i,:] = np.array(npr.choice(num_samples, num_per_seq))

    gs = gridspec.GridSpec(num_seqs*(num_per_seq+1), T)
    gs.update(wspace=.2, hspace=0.0)

    for i in range(num_seqs):
        for t in range(T-1):
            plt.subplot(gs[i*(1+num_per_seq)*T+t])
            plt.imshow(target[ind[i], t].numpy(), interpolation='None')
            plt.gray()
            plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
            if t==0: plt.ylabel('Ground Truth', fontsize=10)

            for m in range(num_per_seq):
                plt.subplot(gs[i*(1+num_per_seq)*T+(m+1)*T+t+1])
                plt.imshow(samples[ind[i], samp_seq_ind[i,m], t].numpy(), interpolation='None')
                plt.gray()
                plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
                if t==0: plt.ylabel('Sample %d'%(m), fontsize=10)

    if savepath is not None: plt.savefig(savepath)

    return plt

def plot_2dviz(data, basis, showplot=True, savepath=None):
    if not showplot:
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        plt.ioff() 
        import matplotlib.gridspec as gridspec
    else:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        plt.ion()
    plt.clf()        
        
    data_trans = np.dot(data.numpy(), basis)
    plt.plot(data_trans[:,0], data_trans[:,1], 'bo')

    if savepath is not None: plt.savefig(savepath)

    return plt

def plot_loss(plt, loss):
    '''
    Plots the loss function during training
    '''

def plot_single_ts(target, samples, savepath):
    # import matplotlib as mpl
    # mpl.use('Agg')
    # import matplotlib.pyplot as plt
    # plt.ioff() 
    # import matplotlib.gridspec as gridspec
    # import imageio
    # plt.clf()


    # gs = gridspec.GridSpec(1, 2)
    # gs.update(wspace=.2, hspace=0.2)
#    images = []
    images_test = []
    
    num_timesteps = target.size(1)
    for t in range(num_timesteps):
        # plt.subplot(gs[0])
        # plt.imshow(target[0,t].numpy(), interpolation='None')
        # plt.title('Original')

        # plt.subplot(gs[1])
        # plt.imshow(samples[0,t].numpy(), interpolation='None')
        # plt.title('Predicted')

        # plt.savefig(savepath+'_'+str(t)+'.png')
        # images.append(imageio.imread(savepath+'_'+str(t)+'.png'))
        images_test.append(np.concatenate((target[0,t].numpy(), samples[0,t].numpy()), axis=1))
#    imageio.mimsave(savepath+'.gif', images, fps=2)
    imageio.mimsave(savepath+'.gif', images_test, fps=2)
        
    
    
