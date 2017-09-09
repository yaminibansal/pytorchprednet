def train(model, optimizer, criterion, input, noise, target):
    '''
    noise: batch_size x num_samples x num_timesteps x noise_dim
    '''
    batch_size = input.size(0)
    num_timesteps = input.size(1)
    optimizer.zero_grad()
    loss = 0
    hidden = model.init_hidden(batch_size)
    

    for t in range(num_timesteps):
        hidden, output = model(input[:,t], hidden, noise[:,:,t], target[:,t].size())
        loss+= criterion.forward(target[:,t], output)
    loss.backward()
    optimizer.step()

    return output, loss.data[0]
