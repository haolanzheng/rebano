import torch
import torch.autograd as autograd
torch.set_default_dtype(torch.float)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def autograd_calculations(x_resid, y_resid, t_resid, P):
    """Compute graidents w.r.t t, x adn xx for the residual data"""
    x_resid = x_resid.requires_grad_().clone()
    y_resid = y_resid.requires_grad_().clone()
    t_resid = t_resid.requires_grad_().clone()
    vor_stream = P(x_resid, y_resid, t_resid).to(device)
    vor    = vor_stream[:, 0:1]
    stream = vor_stream[:, 1:2]
    P_x  = autograd.grad(vor, x_resid, torch.ones_like(vor).to(device), create_graph=True)[0]
    P_xx = autograd.grad(P_x, x_resid, torch.ones_like(P_x).to(device), create_graph=True)[0] 
    P_y  = autograd.grad(vor, y_resid, torch.ones_like(vor).to(device), create_graph=True)[0]
    P_yy = autograd.grad(P_y, y_resid, torch.ones_like(P_y).to(device), create_graph=True)[0]
    P_t = autograd.grad(vor, t_resid, torch.ones_like(vor).to(device), create_graph=True)[0]
    P_x  = P_x.detach()
    P_xx = P_xx.detach()
    P_y  = P_y.detach()
    P_yy = P_yy.detach()
    P_t  = P_t.detach()
    psi_x = autograd.grad(stream, x_resid, torch.ones_like(stream).to(device), create_graph=True)[0]
    psi_y = autograd.grad(stream, y_resid, torch.ones_like(stream).to(device), create_graph=True)[0]
    psi_xx = autograd.grad(psi_x, x_resid, torch.ones_like(psi_x).to(device), create_graph=True)[0]
    psi_yy = autograd.grad(psi_y, y_resid, torch.ones_like(psi_y).to(device), create_graph=True)[0]
    u = psi_y
    v = - psi_x
    U = torch.hstack((u.detach(), v.detach()))
    lap_psi = psi_xx + psi_yy
    lap_psi = lap_psi.detach()
    
    return P_t, P_x, P_xx, P_y, P_yy, U, lap_psi
    
def Pt_nu_lap_vor(nu, P_t, P_xx, P_yy):
    """Pt + nuPxx"""
    lap_P   = torch.add(P_xx, P_yy)
    nu_lap_vor = torch.mul(-nu, lap_P)
    return torch.add(P_t, nu_lap_vor)
