import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import numpy.random as npr

import numpy as np

class GMM(nn.Module):
    def __init__(self, noise_dims, num_comp):
        '''
        noise_dims = Number of noise dimensions
        num_comp = Number of noise components
        '''
        super(GMM, self).__init__()
        self.num_comp = num_comp
        self.noise_dims = noise_dims
        self.means = Parameter(torch.randn(num_comp, noise_dims))

    def forward(self, input, noise, target_size):
        '''
        noise: batch_size x num_samples x noise_dim
        '''
        batch_size = noise.size(0)
        num_samples = noise.size(1)
        comp_ind = Variable(torch.LongTensor(npr.choice(self.num_comp, size=batch_size*num_samples)))
        if torch.cuda.is_available():
            comp_ind = comp_ind.cuda()
        selected_mean = torch.index_select(self.means, 0, comp_ind)
        shifted_noise = 0.1*noise + selected_mean.view(batch_size, num_samples, self.noise_dims)
        shifted_noise = shifted_noise.view((batch_size, num_samples)+target_size[1:])
        return shifted_noise
        
        


class MeanFinder(nn.Module):
    def __init__(self, noise_dims):
        '''
        noise_dims = Number of noise dimensions
        '''
        super(MeanFinder, self).__init__()
        self.mean = Parameter(torch.randn(noise_dims))
        # if sigma is not None:
        #     self.sigma = Parameter(torch.randn(num_dims, num_dims))
        # else:
        #     self.sigma = sigma
        
    def forward(self, input, noise, target_size):
        '''
        noise: batch_size x num_samples x noise_dim
        '''
        batch_size = noise.size(0)
        num_samples = noise.size(1)
        shifted_noise = noise + self.mean.expand_as(noise)
        shifted_noise = shifted_noise.view((batch_size, num_samples)+target_size[1:])
        return shifted_noise

class StochFCDecoder(nn.Module):
    def __init__(self, dec_layer_size):
        '''
        Input Args:
        dec_size: Number of units in each layer of the decoder
        '''
        super(StochFCDecoder, self).__init__()
        self.num_layers = len(dec_layer_size)
        self.dec_layer_size = dec_layer_size

        for layer in range(self.num_layers-1):
            self.__setattr__('linear'+str(layer+1), nn.Linear(self.dec_layer_size[layer], self.dec_layer_size[layer+1]))

        self.act = nn.ReLU()

    def forward(self, input, noise, target_size):
        '''
        target_size will have dim 0 as batch_size
        output should be such that dim 1 is num_samples and all other dims are same as target size
        input: batch_size x dec_layer_size[0]
        noise: batch_size x num_samples x noise_dim
        '''
        num_samples = noise.size(1)
        noise_dim = noise.size(2)
        batch_size = input.size(0)
        output_size = target_size[:1]+(num_samples,)+target_size[1:]
        # goes along batch first
        in_rep = input.repeat(num_samples, 1) # batch_size*num_samples x dec_layer_size[0]

        noise = noise.permute(1, 0, 2).contiguous()
        noise = noise.view(-1, noise_dim) # batch_size*num_samples x noise_dim

        
        dec = dict.fromkeys(np.arange(0, self.num_layers, 1))

        for layer in range(self.num_layers):
            if layer == 0:
                dec[layer] = torch.cat((in_rep, noise), dim=1)
            elif layer == self.num_layers - 1:
                dec[layer] = self.__getattr__('linear'+str(layer))(dec[layer-1])
            else:
                dec[layer] = self.act(self.__getattr__('linear'+str(layer))(dec[layer-1]))

        ndims = list(range(len(output_size)))
        ndims[0]=1
        ndims[1]=0
        ndims = tuple(ndims)
        output = dec[self.num_layers-1].view((num_samples,)+target_size).permute(*ndims)

        return output
