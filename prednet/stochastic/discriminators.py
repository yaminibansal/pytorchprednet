import torch
import torch.nn as nn

class dcgan_netD(nn.Module):
    def __init__(self):
        super(dcgan_netD, self).__init__()

        self.main = nn.Sequential(
            # input is 1 x 20 x 20
            nn.Conv2d(1, 32, 4, 2, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size 32 x 8 x 8
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            # state size 64 x 4 x 4
            nn.Conv2d(64, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)

        return output.view(-1, 1).squeeze(1)
    
    
    
