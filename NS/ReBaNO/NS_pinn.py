import torch
import torch.autograd as autograd
import torch.nn as nn
torch.set_default_dtype(torch.float)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PINN(nn.Module):    
    """PINN of 2d Navier-Stokes problem"""
    def __init__(self, layers, nu, curl_f, omega0, act, omega_true=None):
        super().__init__()
        self.layers = layers
        self.nu     = nu
        self.curl_f = curl_f
        self.omega0 = omega0
        self.omega_true  = omega_true
        self.loss_function = nn.MSELoss(reduction='mean')
        self.linears   = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        
        # self.param_reg = nn.Linear(3, 1, bias=False)
        
        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data)
            nn.init.zeros_(self.linears[i].bias.data)
        
        # nn.init.xavier_normal_(self.param_reg.weight.data)
        self.activation = act
    
    def forward(self, x, y, t):
        a = torch.cat((x, y, t), dim=-1)       
        a = a.float()
        for i in range(0, len(self.layers)-2):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)
        return a
    
    def lossD(self, Nt, xt_data):
        Nxy = xt_data.shape[0] // Nt
        x = xt_data[:, 0:1].clone()
        y = xt_data[:, 1:2].clone()
        t = xt_data[:, 2:3].clone()
        omega = self.forward(x, y, t)
        omega_final = omega[-Nxy:, 0:1]
        return self.loss_function(omega_final, self.omega_true)
    
    def lossR(self, xt_residual):
        """Residual loss function"""
        x_data = xt_residual[:, 0][:, None]
        y_data = xt_residual[:, 1][:, None]
        t_data = xt_residual[:, 2][:, None]
        x_data = x_data.clone().requires_grad_()
        y_data = y_data.clone().requires_grad_()
        t_data = t_data.clone().requires_grad_()
        
        
        vor_stream = self.forward(x_data, y_data, t_data)

        omega = vor_stream[:, 0:1]
        psi   = vor_stream[:, 1:2]
        omega_x = autograd.grad(omega, x_data, torch.ones_like(omega), create_graph=True)[0]
        omega_y = autograd.grad(omega, y_data, torch.ones_like(omega), create_graph=True)[0]
        psi_x = autograd.grad(psi, x_data, torch.ones_like(psi), create_graph=True)[0]
        psi_y = autograd.grad(psi, y_data, torch.ones_like(psi), create_graph=True)[0]
        omega_t = autograd.grad(omega, t_data, torch.ones_like(omega), create_graph=True)[0]
        omega_xx = autograd.grad(omega_x, x_data, torch.ones_like(omega_x), create_graph=True)[0]
        omega_yy = autograd.grad(omega_y, y_data, torch.ones_like(omega_x), create_graph=True)[0]
        psi_xx = autograd.grad(psi_x, x_data, torch.ones_like(psi_x), create_graph=True)[0]
        psi_yy = autograd.grad(psi_y, y_data, torch.ones_like(psi_y), create_graph=True)[0]
                
        f1 = torch.mul(psi_y, omega_x) - torch.mul(psi_x, omega_y)
        f2 = torch.mul(-self.nu, torch.add(omega_xx, omega_yy))
        f3 = torch.add(f1, f2)
        f = torch.add(omega_t, f3)
        f = self.loss_function(f, self.curl_f)
        g = - torch.add(psi_xx, psi_yy)
        g = self.loss_function(omega, g)
        h = self.loss_function(torch.sum(psi, dim=0)/psi.shape[0], torch.tensor([0.]).to(device))
        
        # Nxy = xt_residual.shape[0]//Nt
        # omega_final = omega[(-Nxy):, :]
        
        # h = self.loss_function(omega_final, self.omega_true)
        
        return f + g + h
    
    def lossIC(self, IC_xt):
        """Initial condition loss function"""
        # u0 = self.u0.clone().to(device)
        x_data = IC_xt[:, 0][:, None]
        y_data = IC_xt[:, 1][:, None]
        t_data = IC_xt[:, 2][:, None]
        x_data = x_data.clone().requires_grad_()
        y_data = y_data.clone().requires_grad_()
        t_data = t_data.clone().requires_grad_()
        
        loss_IC = self.loss_function(self.forward(x_data, y_data, t_data)[:, 0:1], self.omega0)
        
        return loss_IC
    
    def lossBC(self, BC_bottom, BC_top, BC_left, BC_right):
        """Periodic boundary condition"""
        x_data_top = BC_top[:, 0][:, None]
        y_data_top = BC_top[:, 1][:, None]
        t_data_top = BC_top[:, 2][:, None]
        x_data_top = x_data_top.clone().requires_grad_()
        y_data_top = y_data_top.clone().requires_grad_()
        t_data_top = t_data_top.clone().requires_grad_()
        
        x_data_bottom = BC_bottom[:, 0][:, None]
        y_data_bottom = BC_bottom[:, 1][:, None]
        t_data_bottom = BC_bottom[:, 2][:, None]
        x_data_bottom = x_data_bottom.clone().requires_grad_()
        y_data_bottom = y_data_bottom.clone().requires_grad_()
        t_data_bottom = t_data_bottom.clone().requires_grad_()
        
        x_data_left = BC_left[:, 0][:, None]
        y_data_left = BC_left[:, 1][:, None]
        t_data_left = BC_left[:, 2][:, None]
        x_data_left = x_data_left.clone().requires_grad_()
        y_data_left = y_data_left.clone().requires_grad_()
        t_data_left = t_data_left.clone().requires_grad_()
        
        x_data_right = BC_right[:, 0][:, None]
        y_data_right = BC_right[:, 1][:, None]
        t_data_right = BC_right[:, 2][:, None]
        x_data_right = x_data_right.clone().requires_grad_()
        y_data_right = y_data_right.clone().requires_grad_()
        t_data_right = t_data_right.clone().requires_grad_()
        
        vor_stream_BC_top = self.forward(x_data_top, y_data_top, t_data_top)
        vor_stream_BC_bottom = self.forward(x_data_bottom, y_data_bottom, t_data_bottom)
        vor_stream_BC_left = self.forward(x_data_left, y_data_left, t_data_left)
        vor_stream_BC_right = self.forward(x_data_right, y_data_right, t_data_right)
        
        loss_BC = self.loss_function(vor_stream_BC_top, vor_stream_BC_bottom) + self.loss_function(vor_stream_BC_left, vor_stream_BC_right)
        # loss_BC2 = self.loss_function(u_BC_x[:N_BC//2, 0], u_BC_x[N_BC//2:, 0])
        # loss_BC  = torch.add(loss_BC1, loss_BC2)
        
        return loss_BC
        

    def loss(self, xt_resid, IC_xt, BC_xt_bottom, BC_xt_top, BC_xt_left, BC_xt_right, using_data=False,
             Nt=None, xt_data=None):
        """Total loss function"""
            
        loss_R    = self.lossR(xt_resid)
        loss_IC   = self.lossIC(IC_xt)
        loss_BC   = self.lossBC(BC_xt_bottom, BC_xt_top, BC_xt_left, BC_xt_right)
        
        if using_data:
            loss_data = self.lossD(Nt, xt_data)
            loss = loss_R + loss_IC + loss_BC + loss_data
        else:
            loss = loss_R + loss_IC + loss_BC
        return  loss
    