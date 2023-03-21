from torch import nn

class SirenLayer(nn.Module):

    def __init__(self, in_size, out_size, first_layer=False, bias=True):
        rand_weights = (torch.rand(size_in)-0.5)*2 #get random numbers between -1 and 1
        self.linear_layer = nn.Linear(in_size, out_size, bias)
        self.w_0 = 1
        if first_layer:
            self.w_0 = 30 #use w_0 of 30 for first layer
        weights = rand_weights * w_0 * torch.sqrt(6)/in_size
        self.linear_layer.weight = weights
    
    def forward(input):
        linear_out = self.linear_layer(input)
        return torch.sin(self.w_0*linear_out)
        

    


class mySiren(nn.Module):
    def __init__(self, in_size=2, out_size=3, hidden_layers=5, hidden_size=256): #these are the valus they used in the paper for most experiments
        layers=[]
        layers.append(SirenLayer(in_size, hidden_size, True)) #first layer
        for i in range(hidden_layers):
            layers.append(SirenLayer(hidden_size, hidden_size))
        layers.append(SirenLayer(hidden_size, out_size))
        model = nn.Sequential(*layers)
    
    def forward(input):
        return self.model(input)





        
