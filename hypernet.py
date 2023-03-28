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
            
            layers=[]
            layers.append(nn.Linear(2700, 256)) #first layer
            last_layer_size = 1
            for dim in params.shape:
                last_layer_size *= dim
            layers.append(nn.Linear(256, last_layer_size))
            layers.append(nn.Unflatten(-1, params.shape))
            self.subnets.append(nn.Sequential(*layers))
            self.keys.append(key)

        
    
    def forward(self, input):

        
        updated_params = OrderedDict()
        print("HELLO")
        for i, subnet in enumerate(self.subnets):
            subnet = self.subnets[i]
            updated_params[self.keys[i]] = subnet(input)
        return updated_params
        