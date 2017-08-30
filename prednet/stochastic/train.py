def train(model, optimizer, criterion, input, noise, target):
    '''
    noise: batch_size x num_samples x noise_dim
    '''
    optimizer.zero_grad()
    output = model(input, noise, target.size())
    assert len(output.size()) == len(target.size()) + 1
    loss = criterion(target, output)
    loss.backward()
    optimizer.step()

    return output, loss.data[0]

