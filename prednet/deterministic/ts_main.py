import argparse

import numpy as np
import numpy.random as npr

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from prednet.deterministic.ts_train import train
from prednet.deterministic.ts_predict import predict
from prednet.deterministic.ts_models import PredNet

from prednet.utils.plotting import plot_det_seq
from prednet.utils.misc import timeSince

import time
import math
import hickle as hkl

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

##########################################################
###### Defining the input arguments for the parser #######
##########################################################

parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
parser.add_argument('--dataset', required=True, help='ball')
parser.add_argument('--train_root', required=True, help='path to training data')
parser.add_argument('--val_root', required=True, help='path to validation data')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=10, help='Batch Size')
