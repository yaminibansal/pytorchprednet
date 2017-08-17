import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.modules.loss import _assert_no_grad

def rbf_kernel_matrix(x, y):
    '''Calculates the L2 distance between the vectors x (N x D) and y (M x D). Returns
    matrix of size N x M '''
    product = torch.mm(x, torch.transpose(y, 0, 1))
    sum_squares = torch.sum(torch.pow(x, 2), 1).expand_as(product) + torch.transpose(torch.sum(torch.pow(y, 2), 1), 0, 1).expand_as(product)
    return sum_squares-2*product

class MMDLossFn(Function): 
    '''The backward fucntion has not been implemented yet because all the operations are performed used valid torch functions 
    which allows backward to be computed across the graph. To make it faster, we could custom implement the backward function'''
    def __init__(self, sigma):
        super(MMDLossFn, self).__init__()
        self.sigma = sigma
        
    def forward(self, input, gen_samples):
        '''Calculate the unbiased MMD Loss with \sigma as a free parameter'''
        N = input.size(0)
        M = gen_samples.size(0)
        k_xx = rbf_kernel_matrix(input, input)
        k_yy = rbf_kernel_matrix(gen_samples, gen_samples)
        k_xy = rbf_kernel_matrix(input, gen_samples)
        if N==1:
            MMD = 0
        else:
            MMD = 1.0/(N*(N-1))*(torch.sum(torch.exp(-k_xx/self.sigma**2))-torch.sum(torch.exp(-torch.diag(k_xx)/self.sigma**2)))
        if M==1:
            MMD += 0
        else:
            MMD += 1.0/(M*(M-1))*(torch.sum(torch.exp(-k_yy/self.sigma**2))-torch.sum(torch.exp(-torch.diag(k_yy)/self.sigma**2)))
        MMD += -2.0/(N*M)*torch.sum(torch.exp(-k_xy/self.sigma**2))
        return MMD

class MMDLoss(nn.Module):
    def __init__(self, sigma):
        '''Instantiate the class with kernel width parameter for the RBF kernel '''
        super(MMDLoss, self).__init__()
        self.sigma = sigma
    
    def forward(self, inputs, samples):
        '''Return the MMD loss function'''
        _assert_no_grad(samples)
        backend_fn = MMDLossFn(self.sigma)
        #backend_fn = getattr(self._backend, type(self).__name__)
        return backend_fn(input, samples)
