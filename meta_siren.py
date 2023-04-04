from torch import nn
import torch
from torchmeta.modules import MetaModule, MetaSequential
from collections import OrderedDict

import re

#borrowed from torchmeta old repos
def get_subdict(dictionary, key=None):
    if dictionary is None:
        return None
    if (key is None) or (key == ''):
        return dictionary
    key_re = re.compile(r'^{0}\.(.+)'.format(re.escape(key)))
    return OrderedDict((key_re.sub(r'\1', k), value) for (k, value)
        in dictionary.items() if key_re.match(k) is not None)

class MetaLinear(nn.Linear, MetaModule):

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)
        weight = params['weight']
        weight = torch.moveaxis(weight, (0,1,2), (0,2,1))
        output = torch.matmul(input,weight)
        output += bias.unsqueeze(-2)
        return output
    
#my own stuff
class MetaSirenLayer(MetaModule):

    def __init__(self, in_size, out_size, first_layer=False, final_layer=False, bias=True):
        super(MetaSirenLayer, self).__init__()
        self.linear_layer = MetaSequential(MetaLinear(in_size, out_size, bias))

        #self.linear_layer.weight.data = weights
        self.final_layer = final_layer
        self.w_0 = 30
    
    def forward(self, input, params):
        linear_out = self.linear_layer(input, get_subdict(params, 'linear_layer'))

        if self.final_layer:
           return linear_out
        else:
            return torch.sin(self.w_0*linear_out)
    
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
        #print("PARANMS", subdict)
        return self.model(input, params=subdict)


        
