import torch
from torch.autograd import Variable

def predict(model, criterion, input, noise, target):
    batch_size = input.size(0)
    num_timesteps = input.size(1)
    loss = 0
    hidden = model.init_hidden(batch_size)
    output_size = target.size()[:1] +(noise.size(1),)+target.size()[1:]
    if torch.cuda.is_available():
        output = Variable(torch.zeros(output_size)).cuda()
    else:
        output = Variable(torch.zeros(output_size))

    for t in range(num_timesteps):
        hidden, output[:,:,t] = model(input[:,t], hidden, noise[:,:,t], target[:,t].size())
        loss+= criterion.forward(target[:,t], output[:,:,t])

    return output, loss.data[0]
