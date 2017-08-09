import torch
import torch.nn as nn
from prednet.utils.ConvLSTM2dCell import ConvLSTM2dCell
from torch.autograd import Variable

class RNN_MMD(nn.Module):
    def __init__(self, M, D_in):
        super(RNN_MMD, self).__init__()
        self.
