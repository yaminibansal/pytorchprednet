def predict(model, criterion, input, target):
    output = model(input, noise, target.size())
    err = criterion(target, output)
    return output, err
