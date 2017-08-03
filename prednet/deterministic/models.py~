import torch.nn as nn
from StochasticPredNet.utils import ConvLSTM2dCell


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

if __name__ == '__main__':

    print('Works')
