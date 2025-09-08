import torch
from torch import tanh
import torch.nn as nn
torch.set_default_dtype(torch.float)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
class P(nn.Module): 
    def __init__(self, layers, weights, bias, act):
        super().__init__()
        self.layers = layers
        self.linears = nn.ModuleList([nn.Linear(self.layers[i], self.layers[i+1]) for i in range(len(self.layers)-1)])
        
        for i in range(len(self.layers)-2):
            if i == 0:
                self.linears[i].weight.data = torch.Tensor(weights[i]).view(layers[1], 1)
                self.linears[i].bias.data   = torch.Tensor(bias[i])
            else:
                self.linears[i].weight.data = torch.Tensor(weights[i])
                self.linears[i].bias.data   = torch.Tensor(bias[i])
        
        self.linears[-1].weight.data = torch.Tensor(weights[-1]).view(self.layers[-1], self.layers[-2])
        self.linears[-1].bias.data = torch.Tensor(bias[-1]).view(-1)
        
        self.activation = act
        
    def forward(self, x):      
        """GPT-PINN Activation Function"""
        a = x.clone()
        for i in range(0, len(self.layers)-2):
            z = self.linears[i](a)
            a = self.activation(z)        
        a = self.linears[-1](a)
        return a