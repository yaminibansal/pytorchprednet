import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np

class FCDecoder(nn.Module):
    def __init__(self, dec_layer_size):
        '''
        Input Args:
        dec_size: Number of units in each layer of the decoder
        '''
        super(FCDecoder, self).__init__()
        self.num_layers = len(dec_layer_size)
        self.dec_layer_size = dec_layer_size

        for layer in range(self.num_layers-1):
            self.__setattr__('linear'+str(layer+1), nn.Linear(self.dec_layer_size[layer], self.dec_layer_size[layer+1]))

        self.act = nn.ReLU()

    def forward(self, input, output_size):
        dec = dict.fromkeys(np.arange(0, self.num_layers, 1))

        for layer in range(self.num_layers):
            if layer == 0:
                dec[layer] = input
            elif layer == self.num_layers - 1:
                dec[layer] = self.__getattr__('linear'+str(layer))(dec[layer-1])
            else:
                dec[layer] = self.act(self.__getattr__('linear'+str(layer))(dec[layer-1]))
                
        output = dec[self.num_layers-1].view(output_size)

        return output

# class DCGANGenerator(nn.Module):
#     def __init__(self, dec_filt_size, )


    
