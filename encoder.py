from torch import nn
class myEncoder(nn.Module):

    def __init__(self):
        super(myEncoder, self).__init__()

        layers=[]
        layers.append(nn.Conv2d(3, 3, 3, stride=1)) #first layer
        layers.append(nn.Flatten())
        self.model = nn.Sequential(*layers)

        
    
    def forward(self, input):

        return self.model(input)
        