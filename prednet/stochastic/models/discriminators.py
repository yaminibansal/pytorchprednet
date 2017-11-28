import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class dcgan_netD(nn.Module):
    def __init__(self):
        super(dcgan_netD, self).__init__()

        self.main = nn.Sequential(
            # input is 1 x 20 x 20
            nn.Conv2d(1, 32, 4, 2, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size 32 x 8 x 8
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            # state size 64 x 4 x 4
            nn.Conv2d(64, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)

        return output.view(-1, 1).squeeze(1)
    
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class dcgan_netD_fc(nn.Module):
    def __init__(self):
        super(dcgan_netD_fc, self).__init__()

        self.main = nn.Sequential(
            # input is 1 x 20 x 20
            nn.Conv2d(1, 32, 4, 2, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size 32 x 8 x 8
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            # state size 64 x 4 x 4
            nn.Conv2d(64, 20, 4, 1, 0, bias=False),
            Flatten(),
            nn.Linear(20, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)

        return output.view(-1, 1).squeeze(1)

class cndtn_dcgan_netD_fc(nn.Module):
    def __init__(self, cndtnl_size):
        super(cndtn_dcgan_netD_fc, self).__init__()

        self.convblock = nn.Sequential(
            # input is 1 x 20 x 20
            nn.Conv2d(1, 32, 4, 2, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size 32 x 8 x 8
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            # state size 64 x 4 x 4
            nn.Conv2d(64, 20, 4, 1, 0, bias=False),
            Flatten()
        )

        self.fcblock = nn.Sequential(
            nn.Linear(20+cndtnl_size, 10),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )

    def forward(self, input, cndtnl, num_samples):
        '''
        input: Image we are trying to disc
        cndtnl: label or additional input we are trying to condition on
        '''
        cndtnl_expanded = cndtnl.repeat(1, num_samples).view(input.size(0), -1)
        img_out = self.convblock(input)
        output = self.fcblock(torch.cat((img_out, cndtnl_expanded), dim=1))

        return output.view(-1, 1).squeeze(1)

class cndtn_dcgan_netD_channel(nn.Module):
    def __init__(self, cndtnl_size, cndtn_label=True):
        '''
        cndtn_label: Bool, tells us if we are conditioning on a label. If false, this is conditioned on a previous image
        cndtnl_size: If image - number of channels, if label - number of categories
        '''
        super(cndtn_dcgan_netD_channel, self).__init__()
        self.cndtn_label = cndtn_label

        self.main = nn.Sequential(
            # input is 1 x 20 x 20
            nn.Conv2d(1+cndtnl_size, 32, 4, 2, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size 32 x 8 x 8
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            # state size 64 x 4 x 4
            nn.Conv2d(64, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input, cndtnl, num_samples):
        '''
        input: Image we are trying to disc
        cndtnl: label or additional input we are trying to condition on
        
        '''
        if self.cndtn_label:
            cndtnl_expanded = cndtnl.repeat(1, num_samples*20*20).view(input.size(0), 20, 20, cndtnl.size(1)).permute(0,3,1,2)
        else:
            cndtnl_expanded = cndtnl.view(cndtnl.size(0), -1).repeat(1, num_samples).view(input.size(0), cndtnl.size(1), cndtnl.size(2), cndtnl.size(3))
        output = self.main(torch.cat((input, cndtnl_expanded), dim=1))
        return output.view(-1, 1).squeeze(1)

class st_FC(nn.Module):
    '''
    Fully connected discriminator for single frame 
    '''
    def __init__(self):
        super(st_FC, self).__init__()
        
        self.main = nn.Sequential(
            nn.Linear(400, 1000),
            nn.ReLU(),
            nn.Linear(1000, 200),
            nn.ReLU(),
            nn.Linear(200, 20),
            nn.ReLU(),
            nn.Linear(20,1),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output

class st_FC_LR(nn.Module):
    '''
    Fully connected discriminator for single frame 
    '''
    def __init__(self):
        super(st_FC_LR, self).__init__()
        
        self.main = nn.Sequential(
            nn.Linear(400, 1000),
            nn.LeakyReLU(0.2, inplace=True),            
            nn.Linear(1000, 2000),
            nn.LeakyReLU(0.2, inplace=True),            
            nn.Linear(2000, 20),
            nn.LeakyReLU(0.2, inplace=True),            
            nn.Linear(20,1),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output


# class cndtn_dcgan_FC(nn.Module):
#     def __init__(self, hid_size):
#         super(cndtn_dcgan, self).__init__()

#         self.main = nn.Sequential(
#             # input is 1 x 20 x 20
#             nn.Conv2d(1, 32, 4, 2, 0, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size 32 x 8 x 8
#             nn.Conv2d(32, 64, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.1, inplace=True),
#             # state size 64 x 4 x 4
#             nn.Conv2d(64, 1, 4, 1, 0, bias=False),
#             nn.Conv2d()

            
            
#             nn.Sigmoid()        
#         )
        

class cndtn_dcgan_dist(nn.Module):
    '''
    Discriminator with hidden state distributed on an image size grid and concatenated with image channels
    '''
    def __init__(self, num_inp_channels):
        super(cndtn_dcgan_dist, self).__init__()
        self.num_inp_channels = num_inp_channels
        self.main = nn.Sequential(
            # input is 1 x 20 x 20
            nn.Conv2d(num_inp_channels, 32, 4, 2, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size 32 x 8 x 8
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            # state size 64 x 4 x 4
            nn.Conv2d(64, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()        
        )

    def forward(self, disc_input_all, hidden_all):

        assert disc_input_all.size(0) == hidden_all.size(1)        
        num_timesteps = disc_input_all.size(1)
        batch_size = disc_input_all.size(0)
        hid_size = hidden_all.size(2)
        im_rows = disc_input_all.size(3)
        im_cols = disc_input_all.size(4)

        output = Variable(torch.zeros(num_timesteps, batch_size))
        if torch.cuda.is_available:
            output = output.cuda()
        
        for t in range(num_timesteps):
            hidden = hidden_all[t].view(batch_size, hid_size, 1, 1).repeat(1, 1, im_rows, im_cols)
            output[t] = self.main(torch.cat( (disc_input_all[:,t],  hidden), dim=1 ))
            
        return output

class cndtn_dcgan_lstm(nn.Module):
    '''
    Discrimiantor with LSTM (similar to Lotter 2015)
    '''
    def __init__(self, disc_hid_size, enc_stack_sizes):
        super(cndtn_dcgan_lstm, self).__init__()
        self.num_enc_layers = len(enc_stack_sizes)
        self.enc_stack_sizes = enc_stack_sizes
        self.disc_hid_size = disc_hid_size

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        for layer in range(self.num_enc_layers-1):
            self.__setattr__('encRecLinear'+str(layer+1), nn.Linear(enc_stack_sizes[layer], enc_stack_sizes[layer+1]))
        self.lstm = nn.LSTMCell(self.enc_stack_sizes[self.num_enc_layers-1], self.disc_hid_size)
            
        for layer in range(self.num_enc_layers-1):
            self.__setattr__('encFrameLinear'+str(layer+1), nn.Linear(enc_stack_sizes[layer], enc_stack_sizes[layer+1]))
        
        self.fc = nn.Linear(2*disc_hid_size, 1)

    def forward(self, input_frames, next_frames):
        '''
        input_frames: frames from time t to t+T-1
        next_frames: real or fake frames from time t+1 to t+T
        '''
        assert input_frames.size(0) == next_frames.size(0)
        batch_size = input_frames.size(0)
        num_timesteps = input_frames.size(1)

        enc_rec = dict.fromkeys(np.arange(0, self.num_enc_layers, 1))
        enc_frame = dict.fromkeys(np.arange(0, self.num_enc_layers, 1))

        hidden = self.init_hidden(batch_size)
        output = Variable(torch.zeros(num_timesteps, batch_size))
        if torch.cuda.is_available:
            output = output.cuda()
        
        for t in range(num_timesteps):

            #Compute encoder stacks
            for layer in range(self.num_enc_layers):
                if layer == 0:
                    enc_rec[layer] = input_frames[:,t].contiguous().view(input_frames[:,t].size(0), -1)
                    enc_frame[layer] = next_frames[:,t].contiguous().view(next_frames[:,t].size(0), -1)
                else:
                    enc_rec[layer] = self.relu(self.__getattr__('encRecLinear'+str(layer))(enc_rec[layer-1]))
                    enc_frame[layer] = self.relu(self.__getattr__('encFrameLinear'+str(layer))(enc_frame[layer-1]))

            hidden = self.lstm(enc_rec[self.num_enc_layers-1], hidden)

            output[t] = self.sigmoid(self.fc(torch.cat((hidden[0], enc_frame[self.num_enc_layers-1]), dim=1)))
        return output

    def init_hidden(self, batch_size):
        if torch.cuda.is_available:
            return Variable(torch.zeros(batch_size, self.disc_hid_size)).cuda(), Variable(torch.zeros(batch_size, self.disc_hid_size)).cuda()
        else:
            return Variable(torch.zeros(batch_size, self.disc_hid_size)), Variable(torch.zeros(batch_size, self.disc_hid_size))
        


        
    
    
