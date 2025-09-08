import torch
from torch import tanh
import torch.autograd as autograd
import torch.nn as nn

torch.set_default_dtype(torch.float)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   
class PINN(nn.Module):   
    """PINN of 1d Poisson equation""" 
    def __init__(self, layers, f_hat, act):
        super().__init__()
        
        self.layers = layers
        self.f_hat  = f_hat
        self.loss_function = nn.MSELoss(reduction='mean').to(device)
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])

        
        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1)
            nn.init.zeros_(self.linears[i].bias.data)  
    
        self.activation = act
    
    def forward(self, x):       
        a = x.float()
        for i in range(0, len(self.layers)-2):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)
        return a
    
    def lossR(self, x_residual):
        """Residual loss function"""
        g = x_residual.clone().requires_grad_()
        u = self.forward(g)
        u_x = autograd.grad(u, g, torch.ones_like(u), create_graph=True)[0]
        u_xx = autograd.grad(u_x, g, torch.ones_like(u_x), create_graph=True)[0]
        u_xx = u_xx[:,[0]]
        return self.loss_function(u_xx, -self.f_hat)
    
    def lossBC(self, BC_x, BC_u):
        """Boundary condition loss function"""
        loss_BC = self.loss_function(self.forward(BC_x), BC_u)
        return loss_BC

    def loss(self, x_residual, BC_x, BC_u):
        """Total loss function"""
        loss_R   = self.lossR(x_residual)
        loss_BC  = self.lossBC(BC_x, BC_u)
        return loss_R + loss_BC 