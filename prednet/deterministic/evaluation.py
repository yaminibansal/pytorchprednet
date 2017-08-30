def predict(model, criterion, input, target):
    output = model(input, target.size())
    err = criterion(output, target)
    return output, err

def predict_timeseries(model, criterion, input, target, hidden=None):
    err = 0
    
    if hidden is None:
        batch_size = input.size(0)
        hidden = model.init_hidden(batch_size)

    for t in range(input.size(1)):
        output, hidden = model(input[:,t], hidden)
        err += criterion(output, target[:,t])

    return output, err
