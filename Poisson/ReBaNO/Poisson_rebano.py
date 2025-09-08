import torch
import torch.nn as nn
torch.set_default_dtype(torch.float)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ReBaNO(nn.Module):
    """ReBaNO of 1d Poisson equation"""
    def __init__(self, layers, P, initial_c, BC_u, f_hat, activation_resid, activation_BC, Pxx):
        super().__init__()
        self.layers     = layers
        self.activation = P
        
        self.loss_function = nn.MSELoss(reduction='mean').to(device)
        self.linears = nn.ModuleList([nn.Linear(layers[1], layers[2], bias=False)])
        
        self.BC_u  = BC_u
        self.f_hat = f_hat
        self.Pxx   = Pxx
        
        self.activation_BC     = activation_BC
        self.activation_resid  = activation_resid
        
        # self.linears[0].weight.data = torch.ones(self.layers[1], self.layers[0])
        self.linears[-1].weight.data = initial_c
        
    def forward(self, datatype=None, test_x=None):
        if test_x is not None:
            a = torch.Tensor().to(device)
            for i in range(0, self.layers[1]):
                a = torch.cat((a, self.activation[i](test_x)), 1)
            final_output = self.linears[-1](a)
            
            return final_output
        
        if datatype == 'residual': # Residual Data Output
            final_output = self.linears[-1](self.activation_resid).to(device)
            return final_output
        
        if datatype == 'boundary': # Boundary Data Output
            final_output = self.linears[-1](self.activation_BC).to(device)
            return final_output

    def lossR(self):
        """Residual loss function"""
        f = self.linears[-1](self.Pxx)
        loss_R = self.loss_function(f, -self.f_hat)
        #print(f)
        return loss_R
    
    def lossBC(self):
        """First initial and both boundary condition loss function"""
        loss_BC = self.loss_function(self.forward(datatype='boundary'), self.BC_u)
        return loss_BC
    
    def lossD(self, u_pinn, u):
        """data loss"""
        u_pred = self.linears[-1](u_pinn)
        u_pred = u_pred.reshape(u.shape)
        return torch.norm(u_pred - u) / torch.norm(u)
            
    def loss(self):
        """Total loss function"""
        loss_R   = self.lossR()
        loss_BC  = self.lossBC()
        return loss_R + loss_BC 
    