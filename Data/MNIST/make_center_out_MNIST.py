import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import os
import hickle as hkl
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms

fname = '/home/ybansal/Documents/Research/pytorchprednet/Data/MNIST/'
# MNIST Dataset 
dataset = dsets.MNIST(root=fname, 
                      train=True, 
                      transform=transforms.ToTensor(),  
                      download=False)

# Data Loader (Input Pipeline)
data_loader = torch.utils.data.DataLoader(dataset=dataset, 
                                          batch_size=1, 
                                          shuffle=True)
data_iter = iter(data_loader)

num_datapoints = 50000
num_valpoints = 10000
num_timesteps = 10 # max possible is 18
train_sequence = torch.zeros((num_timesteps, 1, 64, 64))
val_sequence = torch.zeros((num_timesteps, 1, 64, 64))
traj_train = torch.zeros([num_timesteps, 2])
traj_val = torch.zeros([num_timesteps, 2])
dir_train = torch.zeros([1])
dir_val = torch.zeros([1])

for i in range(num_datapoints):
    print(i)
    x = 18
    y = 18
    dir = npr.choice(np.arange(8), 1)[0] #One of 8 possible directions
    x_dir = 1*(dir==0 or dir==1 or dir==7)+0*(dir==2 or dir==6)-1*(dir==3 or dir==4 or dir==5)
    y_dir = 1*(dir==1 or dir==2 or dir==3)+0*(dir==0 or dir==4)-1*(dir==5 or dir==6 or dir==7)
    im = next(data_iter)[0][0, 0]
    dir_train[0] = dir
    
    for t in range(num_timesteps):
        train_sequence[t, 0, x:x+28, y:y+28] = im
        traj_train[t, 0] = x
        traj_train[t, 1] = y
        x+= x_dir
        y+=y_dir

    tr_storage = {}
    tr_storage['videos'] = train_sequence.numpy()
    tr_storage['trajectories'] = traj_train.numpy()
    tr_storage['directions'] = dir_train.numpy()
    storage_path = '/home/ybansal/Documents/Research/pytorchprednet/Data/MNIST/center_out/train/'+str(i)+'.hkl'
    f = open(storage_path, 'w')
    hkl.dump(tr_storage, f)
    f.close()

for i in range(num_valpoints):
    x = 18
    y = 18
    dir = npr.choice(np.arange(8), 1)[0] #One of 8 possible directions
    x_dir = 1*(dir==0 or dir==1 or dir==7)+0*(dir==2 or dir==6)-1*(dir==3 or dir==4 or dir==5)
    y_dir = 1*(dir==1 or dir==2 or dir==3)+0*(dir==0 or dir==4)-1*(dir==5 or dir==6 or dir==7)
    im = next(data_iter)[0][0, 0]
    dir_val[0] = dir
    
    for t in range(num_timesteps):
        val_sequence[t, 0, x:x+28, y:y+28] = im
        traj_val[t, 0] = x
        traj_val[t, 1] = y        
        x+= x_dir
        y+=y_dir

    val_storage = {}
    val_storage['videos'] = val_sequence.numpy()
    val_storage['trajectories'] = traj_val.numpy()
    val_storage['directions'] = dir_val.numpy()
    storage_path = '/home/ybansal/Documents/Research/pytorchprednet/Data/MNIST/center_out/val/'+str(i)+'.hkl'
    f = open(storage_path, 'w')
    hkl.dump(val_storage, f)
    f.close()




