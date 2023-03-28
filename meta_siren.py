from torch import nn
import torch
from torchmeta.modules import MetaModule, MetaSequential
from collections import OrderedDict

import re

def get_subdict(dictionary, key=None):
    if dictionary is None:
        return None
    if (key is None) or (key == ''):
        return dictionary
    key_re = re.compile(r'^{0}\.(.+)'.format(re.escape(key)))
    return OrderedDict((key_re.sub(r'\1', k), value) for (k, value)
        in dictionary.items() if key_re.match(k) is not None)

#this class is taken from the repo
class BatchLinear(nn.Linear, MetaModule):
    '''A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a
    hypernetwork.'''
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)
        weight = params['weight']

        print("BIAS SHAPE", bias.shappe)
        print("WEIGHT SHAPE", weight.shape)
        print("INPUT SHAPE", input.shape)

        output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
        output += bias.unsqueeze(-2)
        return output

class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)
    
#my own stuff
class MetaSirenLayer(MetaModule):

    def __init__(self, in_size, out_size, first_layer=False, final_layer=False, bias=True):
        super(MetaSirenLayer, self).__init__()
        rand_weights = (torch.rand(out_size, in_size)-0.5)*2 #get random numbers between -1 and 1 (uniform distribution)
        self.linear_layer = MetaSequential(BatchLinear(in_size, out_size, bias), Sine())
        self.w_0 = 30
        if first_layer:
            weights = rand_weights * (1/in_size)
        else:
            weights = rand_weights * ((6/in_size)**0.5) * (1/self.w_0)
        #self.linear_layer.weight.data = weights
        self.final_layer = final_layer
    
    def forward(self, input, params):
        linear_out = self.linear_layer(input, get_subdict(params, 'linear_layer'))
        return linear_out
        #if self.final_layer:
        #    return linear_out
        #else:
            #return torch.sin(self.w_0*linear_out)
    
class myMetaSiren(MetaModule):
    def __init__(self, in_size=2, out_size=3, hidden_layers=3, hidden_size=256): #these are the valus they used in the paper for most experiments
        super(myMetaSiren, self).__init__()
        layers=[]
        layers.append(MetaSequential(MetaSirenLayer(in_size, hidden_size, True))) #first layer
        for i in range(hidden_layers):
            layers.append(MetaSequential(MetaSirenLayer(hidden_size, hidden_size)))
        layers.append(MetaSequential(MetaSirenLayer(hidden_size, out_size, False, True)))
        self.model = MetaSequential(*layers)
    
    def forward(self, input, params=None):

        subdict = get_subdict(params, 'model')
        print("PARANMS", subdict)
        return self.model(input, params=subdict)



# class hello(MetaModule):
#     def __init__(self, in_size=2, out_size=3, hidden_layers=3, hidden_size=256): #these are the valus they used in the paper for most experiments

#         self.model = myMetaSiren(in_size, out_size, hidden_layers, hidden_size)

#     def forward(self, input, params=None):
#         return model_output   

#         print("HI")
#         def get_subdict(dictionary, key=None):
#             if dictionary is None:
#                 return None
#             if (key is None) or (key == ''):
#                 return dictionary
#             key_re = re.compile(r'^{0}\.(.+)'.format(re.escape(key)))
#             return OrderedDict((key_re.sub(r'\1', k), value) for (k, value)
#                 in dictionary.items() if key_re.match(k) is not None)

#         model_output = self.model(input, params=get_subdict(params, 'net'))


        
