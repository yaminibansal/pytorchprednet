import torch
import torch.nn as nn
from torch.autograd import Variable
from prednet.utils.ConvLSTM2dCell import ConvLSTM2dCell

import numpy as np


class ConvLSTMEncDec(nn.Module):
    def __init__(self):
        super(ConvLSTMEncDec, self).__init__()
        self.convEnc1 = nn.Conv2d(1, 50, 3, padding=1)
        self.ReLUEnc1 = nn.Threshold(0.0, 0.0)
        self.poolEnc1 = nn.MaxPool2d(2)
        self.convLSTM1 = ConvLSTM2dCell(50, 100, 3, 3)
        self.convLSTM0 = ConvLSTM2dCell(101, 100, 3, 3)
        self.deconvDec1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.convDec = nn.Conv2d(100, 1, 3, padding=1)
        
    def forward(self, input, R1_hidden, R0_hidden):
        A1_conv = self.convEnc1(input)
        A1_relu = self.ReLUEnc1(A1_conv)
        A1 = self.poolEnc1(A1_relu)
        R1_hidden = self.convLSTM1(A1, R1_hidden) 
        R0_input_topdown = self.deconvDec1(R1_hidden[0])
        R0_input = torch.cat((R0_input_topdown, input), dim=1)
        R0_hidden = self.convLSTM0(R0_input, R0_hidden)
        output = self.convDec(R0_hidden[0])
        return R1_hidden, R0_hidden, output


class PredNet(nn.Module):
    def __init__(self, enc_filt_size, enc_ker_size, hid_filt_size, hid_ker_size, pool_enc_size, dec_ker_size):
        '''
        Input Args:
        enc_filt_size: list containing the number of convolution filters (or channels) for each layer in the encoder stack. Element zero corresponds to the number of input channels (eg: 3 for RGB)
        enc_ker_size: list containing the kernel size for the convolutions for each layer in the encoder stack. It is assumed that the kernel sizes are integers. Element zero is not used. It is assumed that all kernel sizes are odd (for the calculation of the padding)
        pool_enc_size: list containing the pooling layer sizes for each layer in the encoder stack. Element zero is not used.
        hid_filt_size: list containing the number of convoution filters (or channels) for each ConvLSTM layer. 
        hid_ker_size: List containing the kernel size for the convolutions for each layer in the convLSTM. The kernel size for the input convolutions and hidden convolutions are assumed to be equal.
        dec_ker_size: List contatining the kernel size for the convolution step from the hidden state to the prediction for each layer
        '''
        
        super(PredNet, self).__init__()
        self.num_layers = len(hid_filt_size)
        self.hid_filt_size = hid_filt_size
        self.enc_filt_size = enc_filt_size
        self.pool_enc_size = pool_enc_size
        
        for layer in range(self.num_layers):
            if layer==0:
                self.__setattr__('convLSTM'+str(layer), ConvLSTM2dCell(hid_filt_size[layer+1]+enc_filt_size[layer], hid_filt_size[layer], hid_ker_size[layer], hid_ker_size[layer]) )
                self.__setattr__('convDec'+str(layer), nn.Conv2d(hid_filt_size[layer], enc_filt_size[layer], dec_ker_size[layer], padding = (dec_ker_size[layer]-1)/2) ) 
                
            elif layer==self.num_layers -1 :
                self.__setattr__('convEnc'+str(layer), nn.Conv2d(enc_filt_size[layer-1], enc_filt_size[layer], enc_ker_size[layer], padding = (enc_ker_size[layer]-1)/2) )
                self.__setattr__('ReLUEnc'+str(layer), nn.Threshold(0.0, 0.0) )
                self.__setattr__('poolEnc'+str(layer), nn.MaxPool2d(pool_enc_size[layer]) )
                self.__setattr__('convLSTM'+str(layer), ConvLSTM2dCell(enc_filt_size[layer], hid_filt_size[layer], hid_ker_size[layer], hid_ker_size[layer]) )
                self.__setattr__('convDec'+str(layer), nn.Conv2d(hid_filt_size[layer], enc_filt_size[layer], dec_ker_size[layer], padding = (dec_ker_size[layer]-1)/2) )
                self.__setattr__('deconvDec'+str(layer), nn.UpsamplingNearest2d(scale_factor=pool_enc_size[layer]) )                
                
            else:
                self.__setattr__('convEnc'+str(layer), nn.Conv2d(enc_filt_size[layer-1], enc_filt_size[layer], enc_ker_size[layer], padding = (enc_ker_size[layer]-1)/2) )
                self.__setattr__('ReLUEnc'+str(layer),  nn.Threshold(0.0, 0.0))
                self.__setattr__('poolEnc'+str(layer), nn.MaxPool2d(pool_enc_size[layer]) )
                self.__setattr__('convLSTM'+str(layer), ConvLSTM2dCell(hid_filt_size[layer+1]+enc_filt_size[layer], hid_filt_size[layer], hid_ker_size[layer], hid_ker_size[layer]) )
                self.__setattr__('convDec'+str(layer),  nn.Conv2d(hid_filt_size[layer], enc_filt_size[layer], dec_ker_size[layer], padding = (dec_ker_size[layer]-1)/2) )
                self.__setattr__('deconvDec'+str(layer),  nn.UpsamplingNearest2d(scale_factor=pool_enc_size[layer]) )
                

                
    def forward(self, input, hidden):
        '''
        Input Args:
        hidden: Dict containing the hidden states for all the convLSTM layers. The keys of of the dictionary correspond to the layer number and the value is a Variable containing the torch tensor with the hidden state for that layer. Size of each of hidden layer tensor is (batch_size x hid_filt_size[layer_number] x im_size[0]/product(pool_enc_size(:layer_number)) x im_size[1]/product(pool_enc_size(:layer_number)))
        '''

        # Initialize all the dictionaries
        A = dict.fromkeys(np.arange(0, self.num_layers, 1))
        Ahat = dict.fromkeys(np.arange(0, self.num_layers, 1))
        E = dict.fromkeys(np.arange(0, self.num_layers, 1))
        R = dict.fromkeys(np.arange(0, self.num_layers, 1))
        Rin = dict.fromkeys(np.arange(0, self.num_layers, 1))

        # Compute the entire encoder stack
        for layer in range(self.num_layers):
            if layer == 0:
                A[layer] = input
            else:
                A[layer] = self.__getattr__('poolEnc'+str(layer))(self.__getattr__('ReLUEnc'+str(layer))(self.__getattr__('convEnc'+str(layer))(A[layer-1])))

        # Compute the predictions and calculate the errors
        for layer in range(self.num_layers):
            Ahat[layer] = self.__getattr__('convDec'+str(layer))(hidden[layer][0])
            E[layer] = A[layer] - Ahat[layer]

        # Update hidden states through the decoder state
        for layer in np.arange(self.num_layers-1, -1, -1):
            if layer == self.num_layers-1:
                Rin[layer] = E[layer]
                R[layer] = self.__getattr__('convLSTM'+str(layer))(Rin[layer], hidden[layer])
            else:
                Rin[layer] = torch.cat( (self.__getattr__('deconvDec'+ str(layer+1))(R[layer+1][0]), E[layer]), dim=1 )
                R[layer] = self.__getattr__('convLSTM'+str(layer))(Rin[layer], hidden[layer])


        # Produce the output
        output = self.__getattr__('convDec'+str(0))(R[0][0])
        

        return R, output

    def init_hidden(self, batch_size, im_size):
        # Initialize the errors to zero. Need to check the difference between initializing
        # R to zero and updating R with previous step input zero and error zero. This will only
        # be different in the case when the bias is non zero. This will have to be initialized with
        # image size and batch size and filter sizes etc
        
        im_rows, im_cols = im_size
        
        R = dict.fromkeys(np.arange(0, self.num_layers, 1))
        Rin = dict.fromkeys(np.arange(0, self.num_layers, 1))
        E = dict.fromkeys(np.arange(0, self.num_layers, 1))

        for layer in range(self.num_layers):
            if layer is not 0:
                im_rows, im_cols = im_rows/self.pool_enc_size[layer], im_cols/self.pool_enc_size[layer]
            E[layer] = Variable(torch.zeros(batch_size, self.enc_filt_size[layer], im_rows, im_cols))
            R[layer] = Variable(torch.zeros(batch_size, self.hid_filt_size[layer], im_rows, im_cols)), Variable(torch.zeros(batch_size, self.hid_filt_size[layer], im_rows, im_cols))
            if torch.cuda.is_available():
                E[layer] = E[layer].cuda()
                R[layer] = (R[layer][0].cuda(), R[layer][1].cuda())

        for layer in np.arange(self.num_layers-1, 0, -1):
            if layer == self.num_layers-1:
                Rin[layer] = E[layer]
                R[layer] = self.__getattr__('convLSTM'+str(layer))(Rin[layer], R[layer])
            else:
                Rin[layer] = torch.cat((self.__getattr__('deconvDec'+ str(layer))(R[layer+1][0]), E[layer]), dim=1)
                R[layer] = self.__getattr__('convLSTM'+str(layer))(Rin[layer], R[layer])

        return R


class SingleLSTMEncDec(nn.Module):
    def __init__(self, enc_filt_size, enc_ker_size, enc_pool_size, hid_size, dec_filt_size, dec_ker_size, dec_upsample_size, lstm_inp_size):
        super(SingleLSTMEncDec, self).__init__()
        self.num_enc_layers = len(enc_filt_size)
        self.num_dec_layers = len(dec_filt_size)
        self.enc_filt_size = enc_filt_size
        self.enc_ker_size = enc_ker_size
        self.enc_pool_size = enc_pool_size
        self.hid_size = hid_size
        self.dec_filt_size = dec_filt_size
        self.dec_upsample_size = dec_upsample_size

        for layer in range(self.num_enc_layers-1):
            self.__setattr__('convEnc'+str(layer+1), nn.Conv2d(enc_filt_size[layer], enc_filt_size[layer+1], enc_ker_size[layer+1], padding=(enc_ker_size[layer+1]-1)/2 ))
            self.__setattr__('ReLUEnc'+str(layer+1), nn.ReLU())
            self.__setattr__('poolEnc'+str(layer+1), nn.MaxPool2d(enc_pool_size[layer+1]))

        self.lstm = nn.LSTMCell(lstm_inp_size, hid_size)

        for layer in range(self.num_enc_layers-1):
            self.__setattr__('convDec'+str(layer+1), nn.Conv2d(dec_filt_size[layer], dec_filt_size[layer+1], enc_ker_size[layer+1], padding=(enc_ker_size[layer+1]-1)/2 ))
            self.__setattr__('ReLUDec'+str(layer+1), nn.ReLU())
            self.__setattr__('upsampDec'+str(layer+1), nn.UpsamplingNearest2d(scale_factor=dec_upsample_size[layer+1]))

        self.outputlayer = nn.Conv2d(self.dec_filt_size[self.num_dec_layers-1], 1, 3, padding=1)

    def forward(self, input, hidden):

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

        for layer in np.arange(1, self.num_dec_layers, 1):
            decoder_stack[layer] = self.__getattr__('upsampDec'+str(layer))(self.__getattr__('ReLUDec'+str(layer))(self.__getattr__('convDec'+str(layer))(decoder_stack[layer-1])))

        output = self.outputlayer(decoder_stack[self.num_dec_layers-1])

        return hidden, output

    def init_hidden(self, batch_size):
        if torch.cuda.is_available:
            return Variable(torch.zeros(batch_size, self.hid_size)).cuda(), Variable(torch.zeros(batch_size, self.hid_size)).cuda()
        else:
            return Variable(torch.zeros(batch_size, self.hid_size)), Variable(torch.zeros(batch_size, self.hid_size))



