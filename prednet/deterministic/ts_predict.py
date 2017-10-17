from torch.autograd import Variable
import torch

def prednet_predict(model, criterion, input, target):
    batch_size = input.size(0)
    num_timesteps = input.size(1)
    im_size = (input.size(3), input.size(4))
    
    hidden, errors = model.init_hidden(batch_size, im_size)
    
    loss = 0

    output = Variable(torch.zeros(target.size()))

    if torch.cuda.is_available:
        output = output.cuda()

    for t in range(num_timesteps):
        hidden, errors, output[:,t] = model(input[:,t], hidden, errors)
        if criterion is not None:
            loss += criterion.forward(output[:,t], target[:,t])
        else:
            loss = Variable(torch.zeros(1))

    return output, loss.data[0]

            
