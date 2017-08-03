import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable
from torch.nn.modules.utils import _pair
import math

class ConvRNNCellBase(nn.Module):

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)
    
def ConvLSTM2dCellFn(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None, in_stride=(1,1), hid_stride=(1,1), in_padding=0, hid_padding=0, in_dilation=(1,1), hid_dilation=(1,1)):

    hx, cx = hidden
    gates = F.conv2d(input, w_ih, b_ih, in_stride, in_padding, in_dilation) + F.conv2d(hx, w_hh, b_hh, hid_stride, hid_padding, hid_dilation)
    
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
    
    ingate = F.sigmoid(ingate)
    forgetgate = F.sigmoid(forgetgate)
    cellgate = F.tanh(cellgate)
    outgate = F.sigmoid(outgate)
    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * F.tanh(cy)

    return hy, cy

class ConvLSTM2dCell(ConvRNNCellBase):
    """A long short-term memory (LSTM) cell.

    .. math::

        \begin{array}{ll}
        i = \mathrm{sigmoid}(W_{ii} \star x + b_{ii} + W_{hi} \star h + b_{hi}) \\
        f = \mathrm{sigmoid}(W_{if} \star x + b_{if} + W_{hf} \star h + b_{hf}) \\
        g = \tanh(W_{ig} \star x + b_{ig} + W_{hc} h + b_{hg}) \\
        o = \mathrm{sigmoid}(W_{io} \star x + b_{io} + W_{ho} \star h + b_{ho}) \\
        c' = f * c + i * g \\
        h' = o * \tanh(c_t) \\
        \end{array}

        where \star denotes the convolution operator

        Args:
        input_channels (int): Number of channels in the input  
        hidden_channels (int): Number of channels in the hidden state
        in_kernel_size (int or tuple): Size of the convolving kernel for the input, must be odd
        hid_kernel_size (int or tuple): Size of the convolving kernel for the hidden state, must be odd
        in_stride (int or tuple, optional): Stride of the input convolution, Default: 1
        hid_stride (int or tuple, optional): Stride of the hidden convolution, Default: 1
        in_dilation (int or tuple, optional): Spacing between input convolving kernel elements, Default: 1
        hid_dilation (int or tuple, optional): Spacing between hidden convolving kernal elements, Default: 1
        bias (bool, optional): If `False`, then the layer does not use bias weights `b_ih` and `b_hh`. Default: True

    Inputs: input, (h_0, c_0)
        - **input** (batch, in_channels, C_in, H_in): tensor containing input features
        - **h_0** (batch, hidden_channels, C_in, H_in): tensor containing the initial hidden state for each element in the batch.
        - **c_0** (batch, hidden_channels, C_in, H_in): tensor containing the initial cell state for each element in the batch.

    Outputs: h_1, c_1
        - **h_1** (batch, hidden_channels, C_in, H_in): tensor containing the next hidden state for each element in the batch
        - **c_1** (batch, hidden_channels, C_in, H_in): tensor containing the next cell state for each element in the batch

    Attributes:
        weight_ih (Tensor): the learnable input-hidden weights, of shape (hidden_channels, in_channels, kernel_size[0], kernel_size[1])
        weight_hh: the learnable hidden-hidden weights, of shape  (hidden_channels, in_channels, kernel_size[0], kernel_size[1])
        bias_ih: the learnable input-hidden bias, of shape `(hidden_channels)`
        bias_hh: the learnable hidden-hidden bias, of shape `(hidden_channels)`

    """
    def __init__(self, in_channels, hidden_channels, in_kernel_size, hid_kernel_size, in_stride=1, hid_stride=1, in_dilation=1, hid_dilation=1, bias=True):
        super(ConvLSTM2dCell, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.in_kernel_size = _pair(in_kernel_size)
        
        if isinstance(in_kernel_size, int):
            self.in_padding = ((in_kernel_size-1)/2, (in_kernel_size-1)/2)
        else:
            self.in_padding = ((in_kernel_size[0]-1)/2, (in_kernel_size[1]-1)/2)
            
        if isinstance(hid_kernel_size, int):
            self.hid_padding = ((hid_kernel_size-1)/2, (hid_kernel_size-1)/2)
        else:
            self.hid_padding = ((hid_kernel_size[0]-1)/2, (hid_kernel_size[1]-1)/2)
            
        self.hid_kernel_size = _pair(hid_kernel_size)
        self.in_stride = _pair(in_stride)
        self.hid_stride = _pair(hid_stride)
        self.in_dilation = _pair(in_dilation)
        self.hid_dilation = _pair(hid_dilation)
        self.bias = bias

        self.weight_ih = Parameter(torch.Tensor(4 * self.hidden_channels, self.in_channels, *self.in_kernel_size))
        self.weight_hh = Parameter(torch.Tensor(4 * self.hidden_channels, self.hidden_channels, *self.hid_kernel_size))

        if bias:
            self.bias_ih = Parameter(torch.Tensor(4 * self.hidden_channels))
            self.bias_hh = Parameter(torch.Tensor(4 * self.hidden_channels))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.hid_kernel_size:
            n *= k
        stdv = 1. /math.sqrt(n)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
            
    def init_hidden(self, batch_size, im_size):
        im_rows, im_cols = im_size
        return Variable(torch.zeros(batch_size, self.hidden_channels, im_rows, im_cols)), Variable(torch.zeros(batch_size, self.hidden_channels, im_rows, im_cols))
        

    def forward(self, input, hx):
        return ConvLSTM2dCellFn(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
            self.in_stride, self.hid_stride,
            self.in_padding, self.hid_padding,
            self.in_dilation, self.hid_dilation
        )
