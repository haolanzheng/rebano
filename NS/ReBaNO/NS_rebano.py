import torch
import torch.nn as nn
from torch import autograd
torch.set_default_dtype(torch.float)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ReBaNO(nn.Module):
    """ReBaNO of 2d Navier-Stokes problem"""
    def __init__(self, layers, nu, P, initial_c, omega0, curl_f, omega_true,
                 activation_resid, activation_IC, activation_BC_bottom, activation_BC_top, 
                 activation_BC_left, activation_BC_right, Pt_nu_lap_omega, P_vel, P_x, P_y, P_lap_psi):
        super().__init__()
        self.layers = layers
        self.nu = nu
        self.loss_function = nn.MSELoss(reduction='mean')
        self.linears = nn.ModuleList([nn.Linear(layers[1], layers[2], bias=False)])
        self.activation = P
        
        self.activation_resid = activation_resid
        self.activation_IC    = activation_IC
        self.activation_BC_bottom = activation_BC_bottom
        self.activation_BC_top    = activation_BC_top
        self.activation_BC_left   = activation_BC_left
        self.activation_BC_right  = activation_BC_right
        
        self.Pt_nu_lap_omega = Pt_nu_lap_omega
        self.P_vel       = P_vel
        self.P_x = P_x 
        self.P_y = P_y
        self.P_lap_psi = P_lap_psi

        self.omega0 = omega0
        self.curl_f = curl_f
        self.omega_true = omega_true
        
        # nn.init.xavier_uniform_(self.linears[-1].weight.data)
        # self.linears[0].weight.data = torch.ones(self.layers[1], self.layers[0])
        self.linears[-1].weight.data = initial_c
        
    def forward(self, datatype=None, testing=False, test_x=None, test_y=None, test_t=None):
        if testing: # Test Data Forward Pass
            a = torch.Tensor().to(device)
            for i in range(0, self.layers[1]):
                a = torch.cat((a, self.activation[i](test_x, test_y, test_t)[:, 0:1]), 1)
            final_output = self.linears[-1](a)
            return final_output
        
        if datatype == 'residual': # Residual Data Output
            final_output = []
            final_output.append(self.linears[-1](self.activation_resid[:, 0, :]))
            final_output.append(self.linears[-1](self.activation_resid[:, 1, :]))
            return final_output
        
        if datatype == 'initial': # Initial Data Output
            final_output = self.linears[-1](self.activation_IC)
            return final_output
        
        if datatype == 'boundary': # Boundary Data Output
            final_output = []
            final_output.append(self.linears[-1](self.activation_BC_bottom[:, 0, :]))
            final_output.append(self.linears[-1](self.activation_BC_top[:, 0, :]))
            final_output.append(self.linears[-1](self.activation_BC_left[:, 0, :]))
            final_output.append(self.linears[-1](self.activation_BC_right[:, 0, :]))
            final_output.append(self.linears[-1](self.activation_BC_bottom[:, 1, :]))
            final_output.append(self.linears[-1](self.activation_BC_top[:, 1, :]))
            final_output.append(self.linears[-1](self.activation_BC_left[:, 1, :]))
            final_output.append(self.linears[-1](self.activation_BC_right[:, 1, :]))
            return final_output
    
    def lossR(self):
        """Residual loss function"""
            
        vor_stream = self.forward(datatype='residual')
        rebano_omega = vor_stream[0]
        rebano_psi   = vor_stream[1]
        u = self.P_vel[:, 0, :]
        v = self.P_vel[:, 1, :]
        rebano_u = self.linears[-1](u)
        rebano_v = self.linears[-1](v)
        omega_x = self.linears[-1](self.P_x)
        omega_y = self.linears[-1](self.P_y)
        vel_del_omega = torch.add(rebano_u*omega_x, rebano_v*omega_y)
        rebano_Pt_nu_lap_omega = self.linears[-1](self.Pt_nu_lap_omega)
        # ut_vuxx = torch.matmul(self.Pt_nu_P_xx_term, self.linears[-1].weight.data[0][:,None])
        f = torch.add(rebano_Pt_nu_lap_omega, vel_del_omega)
        rebano_lap_psi = self.linears[-1](self.P_lap_psi)
        loss_omega = self.loss_function(f, self.curl_f)
        loss_psi   = self.loss_function(rebano_omega, -rebano_lap_psi)
        loss_mean  = self.loss_function(torch.sum(rebano_psi, dim=0)/rebano_psi.shape[0], torch.tensor([0.]).to(device))
        
        return loss_omega + loss_psi + loss_mean
    
    def lossICBC(self, datatype):
        """First initial loss function"""
        if datatype=='initial':
            # u0 = self.u0.clone().to(device)
            lossIC = self.loss_function(self.forward(datatype), self.omega0)
            return lossIC

            
        elif datatype=='boundary':
            # N_BC = self.activation_BC.shape[0]
            vor_stream_BC = self.forward(datatype)
            omega_BC_b = vor_stream_BC[0]
            omega_BC_t = vor_stream_BC[1]
            omega_BC_l = vor_stream_BC[2]
            omega_BC_r = vor_stream_BC[3]
            
            psi_BC_b = vor_stream_BC[4]
            psi_BC_t = vor_stream_BC[5]
            psi_BC_l = vor_stream_BC[6]
            psi_BC_r = vor_stream_BC[7]
            
            BC_b_values = torch.cat([omega_BC_b, psi_BC_b], dim=-1)
            BC_t_values = torch.cat([omega_BC_t, psi_BC_t], dim=-1)
            BC_l_values = torch.cat([omega_BC_l, psi_BC_l], dim=-1)
            BC_r_values = torch.cat([omega_BC_r, psi_BC_r], dim=-1)
            
            lossBC = self.loss_function(BC_b_values, BC_t_values) \
                            + self.loss_function(BC_l_values, BC_r_values)
            # loss_BC_psi = self.loss_function(psi_BC_b, psi_BC_t) + self.loss_function(psi_BC_l, psi_BC_r)
            
            return lossBC
 
    def loss(self):
        """Total Loss Function"""
        
        loss_R   = self.lossR()
        loss_IC = self.lossICBC(datatype='initial')
        loss_BC  = self.lossICBC(datatype='boundary')
        return loss_R + loss_IC + loss_BC