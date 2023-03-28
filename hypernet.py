from torch import nn
from torchmeta.modules import MetaModule, MetaSequential, MetaLinear
from collections import OrderedDict
import torch

class myHypernet(nn.Module):

    def __init__(self, params_gen):
        super(myHypernet, self).__init__()

        
        self.params_gen = params_gen

        self.subnets = nn.ModuleList()
        self.keys = []
        self.params_shapes = []
        for key, params in self.params_gen:
            print(type(key), key, "model.4" in key)
            #lastlayer=
            if "model.4" in key:
                print("OVER HERE")
                layers=[]
                l = nn.Linear(256, 256)
                l.weight.data = l.weight.data * 0.01
                l.bias.data = (torch.rand(l.bias.shape)-0.5) * 2 * 1/256
                layers.append(l) #first layer
                last_layer_size = 1
                for dim in params.shape:
                    last_layer_size *= dim
                l = nn.Linear(256, last_layer_size)
                l.weight.data = l.weight.data * 0.01
                l.bias.data = (torch.rand(l.bias.shape)-0.5) * 2 * 1/256
                layers.append(l)
                layers.append(nn.Unflatten(-1, params.shape))
                self.subnets.append(nn.Sequential(*layers))
                self.keys.append(key)
            else:
                layers=[]
                layers.append(nn.Linear(256, 256)) #first layer
                last_layer_size = 1
                for dim in params.shape:
                    last_layer_size *= dim
                layers.append(nn.Linear(256, last_layer_size))
                layers.append(nn.Unflatten(-1, params.shape))
                self.subnets.append(nn.Sequential(*layers))
                self.keys.append(key)

        
    
    def forward(self, input):
        updated_params = OrderedDict()
        weights_total = 0
        num_weights = 0
        for i, subnet in enumerate(self.subnets):
            subnet = self.subnets[i]
            weights = subnet(input)
            weights_total += torch.sum(weights**2)
            weights_count = 1
            for dim in weights.shape:
                weights_count *= dim
            num_weights += weights_count
            updated_params[self.keys[i]] = weights
        return updated_params, weights_total/num_weights
        