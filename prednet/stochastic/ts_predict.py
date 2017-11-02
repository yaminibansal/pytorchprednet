import torch
from torch.autograd import Variable

def predict(model, criterion, input, noise, target):
    batch_size = input.size(0)
    num_timesteps = input.size(1)
    loss = 0
    hidden = model.init_hidden(batch_size)
    output_size = target.size()[:1] +(noise.size(1),)+target.size()[1:]
    hidden_size = (num_timesteps,) + hidden[0].size()
    
    if torch.cuda.is_available():
        output = Variable(torch.zeros(output_size)).cuda()
    else:
        output = Variable(torch.zeros(output_size))

    if torch.cuda.is_available():
        hidden_all = Variable(torch.zeros(hidden_size)).cuda()
    else:
        hidden_all = Variable(torch.zeros(hidden_size))

    for t in range(num_timesteps):
        hidden, output[:,:,t] = model(input[:,t], hidden, noise[:,:,t], target[:,t].size())
        if criterion is not None:
            loss+= criterion.forward(target[:,t], output[:,:,t])
        else:
            loss = Variable(torch.zeros(1))
        hidden_all[t] = hidden[0]

    #Permute axis to make 0 batch_size and 1 num_timesteps
    ndims = list(range(len(hidden_size)))
    ndims[0] = 1
    ndims[1] = 0
    ndims = tuple(ndims)
    hidden_all = hidden_all.permute(*ndims)

    return output, hidden_all, loss.data[0]
