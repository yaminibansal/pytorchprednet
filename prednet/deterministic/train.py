def train(model, optimizer, criterion, input, target):
    optimizer.zero_grad()
    output = model(input, target.size())
    assert output.size() == target.size()
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    return output, loss.data[0]
    

def train_timeseries(model, optimizer, criterion, input, target, hidden=None):
    '''
    Assumes that the time axis is 1
    Also assumes output size is target[:,t] size
    Assumes model has an init_hidden function if initial hidden state is not provided
    '''

    optimizer.zero_grad()
    loss = 0

    if hidden is None:
        batch_size = input.size(0)
        hidden = model.init_hidden(batch_size)

    for t in range(input.size(1)):
        output, hidden = model(input[:,t], hidden)
        loss += criterion(output, target[:,t])

    loss.backward()
    optimizer.step()

    return output, loss.data[0]

    

    
    
