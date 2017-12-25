import torch
def prednet_train(model, optimizer, criterion, input, target):
    batch_size = input.size(0)
    num_timesteps = input.size(1)
    im_size = (input.size()[3], input.size(4))

    hidden, errors = model.init_hidden(batch_size, im_size)

    optimizer.zero_grad()

    loss = 0

    for t in range(num_timesteps):
        hidden, errors, output = model(input[:,t], hidden, errors)
#        print('out at %d = %f'%(t, torch.max(output).data.cpu().numpy()))
        if t!=0: loss += criterion.forward(output, target[:,t])/num_timesteps
#        if t!=0: print('Loss at time %d = %f'%(t, loss.cpu().data.numpy()))
        
    loss.backward()

    check = 0
    for param in model.parameters():
        check += torch.max(torch.abs(param.grad))
    print(check.data.cpu().numpy()[0])

    
    optimizer.step()

    return output, loss.data[0]
