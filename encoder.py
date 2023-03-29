from torch import nn
import torch

class ResBlock(nn.Module):
    def __init__(self, channels_in, channels_out=256):
        super(ResBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 5, 1, 2),
            nn.ReLU(),
            nn.Conv2d(channels_out, channels_out, 5, 1, 2),
            nn.ReLU()
        )

        self.myRelU = nn.ReLU()

    def forward(self, x):
        return self.myRelU(self.conv(x) + x)
    
class myEncoder(nn.Module):

    def __init__(self):
        super(myEncoder, self).__init__()

        self.conv_first = nn.Conv2d(3, 128, 3, 1, 1)
        self.relu = nn.ReLU()

        self.cnn = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            ResBlock(256, 256),
            nn.Conv2d(256, 256, 1) 
        )

        self.relu_2 = nn.ReLU()
        self.linear_layer = nn.Linear(1024, 1)


    def forward(self, input):
        o = self.relu_2(self.cnn(self.relu(self.conv_first(input))))
        o = o.view(o.shape[0], 256, -1) #reshape
        o = torch.squeeze(self.linear_layer(o), -1) #remove last layer with dimension 1
        return o
