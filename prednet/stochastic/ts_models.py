import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np                

class StochLSTMEncDec(nn.Module):
    def __init__(self, enc_filt_size, enc_ker_size, enc_pool_size, hid_size, dec_filt_size, dec_ker_size, dec_upsample_size, lstm_inp_size, num_noise_dims):
        super(StochLSTMEncDec, self).__init__()
        self.num_enc_layers = len(enc_filt_size)
        self.num_dec_layers = len(dec_filt_size)
        self.enc_filt_size = enc_filt_size
        self.enc_ker_size = enc_ker_size
        self.enc_pool_size = enc_pool_size
        self.hid_size = hid_size
        self.dec_filt_size = dec_filt_size
        self.dec_upsample_size = dec_upsample_size
        self.num_noise_dims = num_noise_dims

        for layer in range(self.num_enc_layers-1):
            self.__setattr__('convEnc'+str(layer+1), nn.Conv2d(enc_filt_size[layer], enc_filt_size[layer+1], enc_ker_size[layer+1], padding=(enc_ker_size[layer+1]-1)/2 ))
            self.__setattr__('ReLUEnc'+str(layer+1), nn.ReLU())
            self.__setattr__('poolEnc'+str(layer+1), nn.MaxPool2d(enc_pool_size[layer+1]))

        self.lstm = nn.LSTMCell(lstm_inp_size, hid_size)

        for layer in range(self.num_enc_layers-1):
            if layer==0:
                self.__setattr__('convDec'+str(layer+1), nn.Conv2d(dec_filt_size[layer]+self.num_noise_dims, dec_filt_size[layer+1], enc_ker_size[layer+1], padding=(enc_ker_size[layer+1]-1)/2 ))
            else:
                self.__setattr__('convDec'+str(layer+1), nn.Conv2d(dec_filt_size[layer], dec_filt_size[layer+1], enc_ker_size[layer+1], padding=(enc_ker_size[layer+1]-1)/2 ))
            self.__setattr__('ReLUDec'+str(layer+1), nn.ReLU())
            self.__setattr__('upsampDec'+str(layer+1), nn.UpsamplingNearest2d(scale_factor=dec_upsample_size[layer+1]))

        self.outputlayer = nn.Conv2d(self.dec_filt_size[self.num_dec_layers-1], 1, 3, padding=1)

    def forward(self, input, hidden, num_samples):

        batch_size = input.size()[0]

        encoder_stack = dict.fromkeys(np.arange(0, self.num_enc_layers, 1))
        decoder_stack = dict.fromkeys(np.arange(0, self.num_dec_layers, 1))

        for layer in range(self.num_enc_layers):
            if layer == 0:
                encoder_stack[layer] = input
            else:
                encoder_stack[layer] = self.__getattr__('poolEnc'+str(layer))(self.__getattr__('ReLUEnc'+str(layer))(self.__getattr__('convEnc'+str(layer))(encoder_stack[layer-1])))
                             
        encoded_shape = encoder_stack[self.num_enc_layers-1].size()
        assert self.dec_filt_size[0]*encoded_shape[2]*encoded_shape[3] == self.hid_size
        
        lstm_in = encoder_stack[self.num_enc_layers-1].view(encoded_shape[0], -1)

        hidden = self.lstm(lstm_in, hidden)
        decoder_stack[0] = hidden[0].view(encoded_shape[0], self.dec_filt_size[0], encoded_shape[2], encoded_shape[3])

        #Repeat num_samples times
        decoder_stack[0] = decoder_stack[0].repeat(num_samples, 1, 1, 1)
        #print('1:', decoder_stack[0].size())
        #print('2:', decoder_stack[0][0])
        #print('3:', decoder_stack[0][10])
        #print('chk:', decoder_stack[0].view(10, 5, 128, 5, 5)[0, 0])
        #print('chk:', decoder_stack[0].view(10, 5, 128, 5, 5)[0, 1])
        
        #Generate noise samples of the appropriate dimensions
        noise = torch.randn(batch_size*num_samples, self.num_noise_dims)
        noise = Variable( noise.repeat(encoded_shape[2]*encoded_shape[3], 1).view(encoded_shape[2], encoded_shape[3], batch_size*num_samples, self.num_noise_dims).permute(2, 3, 0, 1))
        #print('4:', noise.size())
        #print('5:', noise[:,:,0,0])
        #print('6:', noise[:,:,0,1])
        #print('7:', noise[:,:,1,0])
        #print('8:', noise[:,:,1,1])

        if torch.cuda.is_available():
            noise = noise.cuda()

        decoder_stack[0] = torch.cat((decoder_stack[0], noise), dim=1)

        for layer in np.arange(1, self.num_dec_layers, 1):
            decoder_stack[layer] = self.__getattr__('upsampDec'+str(layer))(self.__getattr__('ReLUDec'+str(layer))(self.__getattr__('convDec'+str(layer))(decoder_stack[layer-1])))

        output = self.outputlayer(decoder_stack[self.num_dec_layers-1])
        output = output.view(batch_size, num_samples, output.size()[1], output.size()[2], output.size()[3])

        #print('9:', output.size())

        return hidden, output

    def init_hidden(self, batch_size):
        if torch.cuda.is_available:
            return Variable(torch.zeros(batch_size, self.hid_size)).cuda(), Variable(torch.zeros(batch_size, self.hid_size)).cuda()
        else:
            return Variable(torch.zeros(batch_size, self.hid_size)), Variable(torch.zeros(batch_size, self.hid_size))

class StochLSTMFCEncDec(nn.Module):
    def __init__(self, enc_stack_sizes, dec_stack_sizes, hid_size):
        super(StochLSTMFCEncDec, self).__init__()
        self.num_enc_layers = len(enc_stack_sizes)
        self.num_dec_layers = len(dec_stack_sizes)
        self.hid_size = hid_size
        self.enc_stack_sizes = enc_stack_sizes
        self.dec_stack_sizes = dec_stack_sizes

        self.act = nn.ReLU()
        
        for layer in range(self.num_enc_layers-1):
            self.__setattr__('encLinear'+str(layer+1), nn.Linear(enc_stack_sizes[layer], enc_stack_sizes[layer+1]))

            self.lstm = nn.LSTMCell(self.enc_stack_sizes[self.num_enc_layers-1], self.hid_size)
            
        for layer in range(self.num_dec_layers-1):
            self.__setattr__('decLinear'+str(layer+1), nn.Linear(dec_stack_sizes[layer], dec_stack_sizes[layer+1]))
    


    def forward(self, input, hidden, noise, target_size):
        '''
        target_size will have dim 0 as batch_size
        output should be such that dim 1 is num_samples and all other dims are same as target size
        input: batch_size x enc_layer_size[0]
        hidden: batch_size x hid_size
        noise: batch_size x num_samples x noise_dim

        '''
        num_samples = noise.size(1)
        noise_dim = noise.size(2)
        output_size = target_size[:1]+(num_samples,) + target_size[1:]
        assert self.hid_size + noise_dim == self.dec_stack_sizes[0]
        
        ###############################
        #### Compute encoder stack ####
        ###############################
        encoder_stack = dict.fromkeys(np.arange(0, self.num_enc_layers, 1))
        for layer in range(self.num_enc_layers):
            if layer == 0:
                encoder_stack[layer] = input.contiguous().view(input.size(0), -1)
            else:
                encoder_stack[layer] = self.act(self.__getattr__('encLinear'+str(layer))(encoder_stack[layer-1]))

        ############################
        #### Compute LSTM state ####
        ############################
        hidden = self.lstm(encoder_stack[self.num_enc_layers-1], hidden)
        

        ########################################
        #### Prepare noise and hidden state ####
        ########################################
        hidden_rep = hidden[0].repeat(num_samples, 1)
        noise = noise.permute(1, 0, 2).contiguous()
        noise = noise.view(-1, noise_dim)

        ###############################
        #### Compute decoder stack ####
        ###############################
        decoder_stack = dict.fromkeys(np.arange(0, self.num_dec_layers), 1)

        for layer in range(self.num_dec_layers):
            if layer == 0:
                decoder_stack[layer] = torch.cat((hidden_rep, noise), dim=1)
            elif layer == self.num_dec_layers -1:
                decoder_stack[layer] = self.__getattr__('decLinear'+str(layer))(decoder_stack[layer-1])
            else:
                decoder_stack[layer] = self.act(self.__getattr__('decLinear'+str(layer))(decoder_stack[layer-1]))

        ndims = list(range(len(output_size)))
        ndims[0]=1
        ndims[1]=0
        ndims = tuple(ndims)
        output = decoder_stack[self.num_dec_layers-1].view((num_samples,)+target_size).permute(*ndims)

        return hidden, output

    def init_hidden(self, batch_size):
        if torch.cuda.is_available:
            return Variable(torch.zeros(batch_size, self.hid_size)).cuda(), Variable(torch.zeros(batch_size, self.hid_size)).cuda()
        else:
            return Variable(torch.zeros(batch_size, self.hid_size)), Variable(torch.zeros(batch_size, self.hid_size))
