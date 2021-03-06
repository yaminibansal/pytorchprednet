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
    def __init__(self, dec_layer_size, pixel_min, pixel_max):
        '''
        Input Args:
        dec_size: Number of units in each layer of the decoder
        '''
        super(StochFCDecoder, self).__init__()
        self.num_layers = len(dec_layer_size)
        self.dec_layer_size = dec_layer_size
        self.pixel_max = pixel_max
        self.pixel_min = pixel_min

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
        batch_size = target_size[0]
        output_size = target_size[:1]+(num_samples,)+target_size[1:]
        # goes along batch first
        if input is not None:
            in_rep = input.repeat(num_samples, 1) # batch_size*num_samples x dec_layer_size[0]

        noise = noise.permute(1, 0, 2).contiguous()
        noise = noise.view(-1, noise_dim) # batch_size*num_samples x noise_dim

        
        dec = dict.fromkeys(np.arange(0, self.num_layers, 1))

        for layer in range(self.num_layers):
            if layer == 0:
                if input is not None:
                    dec[layer] = torch.cat((in_rep, noise), dim=1)
                else:
                    dec[layer] = noise
            elif layer == self.num_layers - 1:
                dec[layer] = self.__getattr__('linear'+str(layer))(dec[layer-1])
            else:
                dec[layer] = self.act(self.__getattr__('linear'+str(layer))(dec[layer-1]))

        ndims = list(range(len(output_size)))
        ndims[0]=1
        ndims[1]=0
        ndims = tuple(ndims)
        output = dec[self.num_layers-1].view((num_samples,)+target_size).permute(*ndims)
        if self.pixel_max is not None or self.pixel_min is not None:
            output = torch.clamp(output, min=self.pixel_min, max=self.pixel_max)
        return output


class StochFCEncDec(nn.Module):
    def __init__(self, enc_layer_size, dec_layer_size, pixel_min, pixel_max):
        '''
        Input Args:
        enc_size: Number of units in each layer of the encoder
        dec_size: Number of units in each layer of the decoder
        '''
        super(StochFCEncDec, self).__init__()
        self.num_enc_layers = len(enc_layer_size)
        self.enc_layer_size = enc_layer_size
        self.num_dec_layers = len(dec_layer_size)
        self.dec_layer_size = dec_layer_size
        self.pixel_max = pixel_max
        self.pixel_min = pixel_min

        for layer in range(self.num_enc_layers-1):
            self.__setattr__('enclinear'+str(layer+1), nn.Linear(self.enc_layer_size[layer], self.enc_layer_size[layer+1]))

        for layer in range(self.num_dec_layers-1):
            self.__setattr__('declinear'+str(layer+1), nn.Linear(self.dec_layer_size[layer], self.dec_layer_size[layer+1]))

        self.act = nn.ReLU()

    def forward(self, input, noise, target_size):
        '''
        target_size will have dim 0 as batch_size
        output should be such that dim 1 is num_samples and all other dims are same as target size
        input: batch_size x num_channels x num_rows x num_cols
        noise: batch_size x num_samples x noise_dim
        '''
        num_samples = noise.size(1)
        noise_dim = noise.size(2)
        batch_size = target_size[0]
        output_size = target_size[:1]+(num_samples,)+target_size[1:]

        enc = dict.fromkeys(np.arange(0, self.num_enc_layers, 1))
        # Compute encoding
        for layer in range(self.num_enc_layers):
            if layer == 0:
                enc[layer] = input.view(input.size(0), -1)
            else:
                enc[layer] = self.act(self.__getattr__('enclinear'+str(layer))(enc[layer-1]))

        hid_state = enc[self.num_enc_layers-1]
        #hid_rep = hid_state.repeat(1, num_samples).view(batch_size*num_samples, hid_state.size(1)) # batch_size*num_samples x dec_layer_size[0]
        hid_rep = hid_state.repeat(num_samples, 1)

        noise = noise.permute(1, 0, 2).contiguous()
        noise = noise.view(-1, noise_dim) # batch_size*num_samples x noise_dim

        
        dec = dict.fromkeys(np.arange(0, self.num_dec_layers, 1))

        for layer in range(self.num_dec_layers):
            if layer == 0:
                dec[layer] = torch.cat((hid_rep, noise), dim=1)
            elif layer == self.num_dec_layers - 1:
                dec[layer] = self.__getattr__('declinear'+str(layer))(dec[layer-1])
            else:
                dec[layer] = self.act(self.__getattr__('declinear'+str(layer))(dec[layer-1]))

        ndims = list(range(len(output_size)))
        ndims[0]=1
        ndims[1]=0
        ndims = tuple(ndims)
        output = dec[self.num_dec_layers-1].view((num_samples,)+target_size).permute(*ndims)
        if self.pixel_max is not None or self.pixel_min is not None:
            output = torch.clamp(output, min=self.pixel_min, max=self.pixel_max)
        return output

    def get_hidden_state(self, input):
        enc = dict.fromkeys(np.arange(0, self.num_enc_layers, 1))
        # Compute encoding
        for layer in range(self.num_enc_layers):
            if layer == 0:
                enc[layer] = input.view(input.size(0), -1)
            else:
                enc[layer] = self.act(self.__getattr__('enclinear'+str(layer))(enc[layer-1]))

        hid_state = enc[self.num_enc_layers-1]
        return hid_state

class StochFCConvEncDec(nn.Module):
    def __init__(self, hid_size, dec_layer_size, pixel_min, pixel_max):
        '''
        Input Args:
        enc_size: Number of units in each layer of the encoder
        dec_size: Number of units in each layer of the decoder
        '''
        super(StochFCConvEncDec, self).__init__()
        self.num_dec_layers = len(dec_layer_size)
        self.dec_layer_size = dec_layer_size
        self.pixel_max = pixel_max
        self.pixel_min = pixel_min

        self.convblock = nn.Sequential(
            # input is 1 x 20 x 20
            nn.Conv2d(1, 32, 4, 2, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size 32 x 8 x 8
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            # state size 64 x 4 x 4
            nn.Conv2d(64, hid_size, 4, 1, 0, bias=False),
            #nn.Sigmoid()
        )        
        
        for layer in range(self.num_dec_layers-1):
            self.__setattr__('declinear'+str(layer+1), nn.Linear(self.dec_layer_size[layer], self.dec_layer_size[layer+1]))

        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, input, noise, target_size):
        '''
        target_size will have dim 0 as batch_size
        output should be such that dim 1 is num_samples and all other dims are same as target size
        input: batch_size x num_channels x num_rows x num_cols
        noise: batch_size x num_samples x noise_dim
        '''
        num_samples = noise.size(1)
        noise_dim = noise.size(2)
        batch_size = target_size[0]
        output_size = target_size[:1]+(num_samples,)+target_size[1:]

        hid_state = self.convblock(input).squeeze()
        
        hid_rep = hid_state.repeat(1, num_samples).view(batch_size*num_samples, hid_state.size(1)) # batch_size*num_samples x dec_layer_size[0]

        noise = noise.permute(1, 0, 2).contiguous()
        noise = noise.view(-1, noise_dim) # batch_size*num_samples x noise_dim
        
        dec = dict.fromkeys(np.arange(0, self.num_dec_layers, 1))

        for layer in range(self.num_dec_layers):
            if layer == 0:
                    dec[layer] = torch.cat((hid_rep, noise), dim=1)
            elif layer == self.num_dec_layers - 1:
                dec[layer] = self.__getattr__('declinear'+str(layer))(dec[layer-1])
            else:
                dec[layer] = self.act(self.__getattr__('declinear'+str(layer))(dec[layer-1]))

        ndims = list(range(len(output_size)))
        ndims[0]=1
        ndims[1]=0
        ndims = tuple(ndims)
        output = dec[self.num_dec_layers-1].view((num_samples,)+target_size).permute(*ndims)
        if self.pixel_max is not None or self.pixel_min is not None:
            output = torch.clamp(output, min=self.pixel_min, max=self.pixel_max)
        return output

    def get_hidden_state(self, input):
        hid_state = self.convblock(input).squeeze()
        return hid_state



    
#class dcgan_netG(nn.Module):
#    def __init__(self):

#    def forward():

        
