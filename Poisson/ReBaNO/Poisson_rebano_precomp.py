import torch
from torch import cos
import torch.autograd as autograd
torch.set_default_dtype(torch.float)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def autograd_calculations(x_resid, P):
    """Compute gradients w.r.t xx for the residual data"""
    x_resid = x_resid.to(device).requires_grad_()
    Pi = P(x_resid).to(device)
    P_x = autograd.grad(Pi, x_resid, torch.ones(x_resid.shape[0], 1).to(device), create_graph=True)[0]
    P_xx = autograd.grad(P_x, x_resid, torch.ones(x_resid.shape[0], 1).to(device), create_graph=True)[0] 

    P_xx = P_xx[:,[0]].detach()
    
    return P_xx
