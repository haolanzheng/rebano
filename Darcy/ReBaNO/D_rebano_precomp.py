import torch
import torch.autograd as autograd
torch.set_default_dtype(torch.float)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def autograd_calculations(Nelementx, Nelementy, N_q, X_flat, Y_flat, P):
    """Compute graidents w.r.t t, x adn xx for the residual data"""   
    u_pinn_all = P(X_flat, Y_flat)  
    u_x_all = autograd.grad(u_pinn_all, X_flat, torch.ones_like(u_pinn_all), create_graph=True)[0]
    u_y_all = autograd.grad(u_pinn_all, Y_flat, torch.ones_like(u_pinn_all), create_graph=True)[0]
    u_x = u_x_all.reshape(Nelementx-1, Nelementy-1, N_q).detach()
    u_y = u_y_all.reshape(Nelementx-1, Nelementy-1, N_q).detach()
    
    return u_x, u_y
