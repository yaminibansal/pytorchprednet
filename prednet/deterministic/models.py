import torch
import torch.nn as nn
from prednet.utils.ConvLSTM2dCell import ConvLSTM2dCell


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
        enc_ker_size: list containing the kernel size for the convolutions for each layer in the encoder stack. Element zero is not used. It is assumed that all kernel sizes are odd (for the calculation of the padding)
        pool_enc_size: list containing the pooling layer sizes for each layer in the encoder stack. Element zero is not used.
        hid_filt_size: list containing the number of convoution filters (or channels) for each ConvLSTM layer. 
        hid_ker_size: List containing the kernel size for the convolutions for each layer in the convLSTM. The kernel size for the input convolutions and hidden convolutions are assumed to be equal.
        dec_ker_size: List contatining the kernel size for the convolution step from the hidden state to the prediction for each layer
        '''
        
        super(PredNet, self).__init__()
        self.num_layers = hidden_filters.shape[0]
        self.hid_filt_size = hid_filt_size
        self.enc_filt_size = enc_filt_size
        self.pool_enc_size = pool_enc_size
        
        for layer in range(self.num_layers):
            if layer==0:
                self.__setattr__('convLSTM'+str(layer)) = nn.Conv2d(hid_filt_size[layer+1]+enc_filt_size[layer], hid_filt_size[layer], hid_ker_size[layer], hid_ker_size[layer])
                self.__setattr__('convDec'+str(layer)) = nn.Conv2d(hid_filt_size[layer], enc_filt_size[layer], dec_ker_size[layer])
                
            elif layer==num_layers -1 :
                self.__setattr__('convEnc'+str(layer)) = nn.Conv2d(enc_filt_size[layer], enc_filt_size[layer+1], enc_ker_size[layer], padding = (enc_ker_size[layer]-1)/2)
                self.__setattr__('ReLUEnc'+str(layer)) = nn.Threshold(0.0, 0.0)
                self.__setattr__('poolEnc'+str(layer)) = nn.MaxPool2d(pool_enc_size[layer])
                self.__setattr__('convLSTM'+str(layer)) = nn.Conv2d(hid_filt_size[layer+1]+enc_filt_size[layer], hid_filt_size[layer], hid_ker_size[layer], hid_ker_size[layer])
                self.__setattr__('convDec'+str(layer)) = nn.Conv2d(hid_filt_size[layer], enc_filt_size[layer], dec_ker_size[layer])
                self.__setattr__('deconvDec'+str(layer)) = nn.UpsamplingNearest2d(scale_factor=pool_enc_size[layer])
                
            else:
                self.__setattr__('convEnc'+str(layer)) = nn.Conv2d(enc_filt_size[layer], enc_filt_size[layer+1], enc_ker_size[layer], padding = (enc_ker_size[layer]-1)/2)
                self.__setattr__('ReLUEnc'+str(layer)) = nn.Threshold(0.0, 0.0)
                self.__setattr__('poolEnc'+str(layer)) = nn.MaxPool2d(pool_enc_size[layer])
                self.__setattr__('convLSTM'+str(layer)) = nn.Conv2d(enc_filt_size[layer], hid_filt_size[layer], hid_ker_size[layer], hid_ker_size[layer])
                self.__setattr__('convDec'+str(layer)) = nn.Conv2d(hid_filt_size[layer], enc_filt_size[layer], dec_ker_size[layer])
                self.__setattr__('deconvDec'+str(layer)) = nn.UpsamplingNearest2d(scale_factor=pool_enc_size[layer])
                
    def forward(self, input, hidden):
        '''
        Input Args:
        hidden: Dict containing the hidden states for all the convLSTM layers. The keys of of the dictionary correspond to the layer number.
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
            elif layer == 1:
                A[layer] = self.__getattr__('poolEnc'+str(layer))(self.__getattr__('ReLUEnc'+str(layer))(self.__getattr__('convEnc'+str(layer))(input)))

        # Compute the predictions and calculate the errors
        for layer in range(self.num_layers):
            Ahat[layer] = self.__getattr__('convDec'+str(layer))(hidden[layer][0])
            E[layer] = A[layer] - Ahat[layer]

        # Update hidden states through the decoder state
        for layer in np.arange(self.num_layers-1, 0, -1):
            if layer == num_layers-1:
                Rin[layer] = E[layer]
                R[layer] = self.__getattr__('convLSTM'+str(layer))(Rin[layer], hidden[layer][0])
            else:
                Rin[layer] = torch.cat((self.__getattr__('deconvDec', str(layer))(R[l]), E[layer]), dim=1)
                R[layer] = self.__getattr__('convLSTM'+str(layer))(Rin[layer], hidden[layer][0])

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

        for layer in range(self.num_layers):
            if layer is not 0:
                im_rows, im_cols = im_rows/pool_enc_size[layer], im_cols/pool_enc_size[layer]
            E[layer] = Variable(torch.zeros(batch_size, self.enc_filt_size[layer], im_rows, im_cols))
            hidden[layer] = Variable(torch.zeros(batch_size, self.hid_filt_size[layer], im_rows, im_cols)), Variable(torch.zeros(batch_size, self.hid_filt_size, im_rows, im_cols))

        for layer in np.arange(self.num_layers-1, 0, -1):
            if layer == num_layers-1:
                Rin[layer] = E[layer]
                R[layer] = self.__getattr__('convLSTM'+str(layer))(Rin[layer], hidden[layer][0])
            else:
                Rin[layer] = torch.cat((self.__getattr__('deconvDec', str(layer))(R[l]), E[layer]), dim=1)
                R[layer] = self.__getattr__('convLSTM'+str(layer))(Rin[layer], hidden[layer][0])

        return R


