def prednet_train(model, optimizer, criterion, input, target):
    batch_size = input.size(0)
    num_timesteps = input.size(1)
    im_size = (input.size()[3], input.size(4))

    hidden, errors = model.init_hidden(batch_size, im_size)

    optimizer.zero_grad()

    loss = 0

    for t in range(num_timesteps):
        hidden, errors, output = model(input[:,t], hidden, errors)
        loss += criterion.forward(output, target[:,t])
        
    loss.backward()
    optimizer.step()

    return output, loss.data[0]
