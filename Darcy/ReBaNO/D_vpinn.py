import torch
from torch import nn
from torch import autograd
import numpy as np
from GaussJacobiQuadRule_V3 import Jacobi, DJacobi

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float)
torch.manual_seed(0)

class VPINN(nn.Module):
    def __init__(self, 
                 layers, 
                 act, 
                 a_func,        # input a(x,y)
                 f_func,        # f(x,y) = 1
                 N_elementx,    # number of grid points in x-axis
                 N_elementy,    # number of grid points in y-axis
                 Ntest_func,    # number of total test functions
                 X_quad,        # quadrature points
                 W_quad,        # quadrature weights
                 gridx,         # x grid
                 gridy,         # y grid
                 xy_BC_bottom,  # collocation points on bottom boundary (y=0)
                 xy_BC_top,     # collocation points on top boundary (y=1)
                 xy_BC_left,    # collocation points on left boundary (x=0)
                 xy_BC_right,   # collocation points on right boundary (x=1)
                 ):
        super(VPINN, self).__init__()
        self.to(device)
        self.layers = layers
        self.activation = act
        
        # PDE data and quadrature
        self.a_func  = a_func
        self.f_func = f_func
        
        self.Nelementx = N_elementx 
        self.Nelementy = N_elementy
        
        self.Ntest_func = Ntest_func
        
        self.x_quad  = X_quad[:, 0:1]
        self.y_quad  = X_quad[:, 1:2]
        self.w_quad  = W_quad
        
        self.gridx = gridx 
        self.gridy = gridy
        
        self.dx = self.gridx[1] - self.gridx[0]
        self.dy = self.gridy[1] - self.gridy[0]
        
        # Boundary collocation points
        self.xy_BC_bottom = xy_BC_bottom
        self.xy_BC_top    = xy_BC_top
        self.xy_BC_left   = xy_BC_left
        self.xy_BC_right  = xy_BC_right
        
        self.linears   = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
        
        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data)
            nn.init.zeros_(self.linears[i].bias.data)
        
        # Prepare storage for element quadrature nodes & Jacobians
        self.x_quad_elem = torch.zeros(self.Nelementx-1, self.x_quad.shape[0], requires_grad=True).to(device)
        self.y_quad_elem = torch.zeros(self.Nelementy-1, self.y_quad.shape[0], requires_grad=True).to(device)
        
        self.jacobian_x = torch.zeros(self.Nelementx-1, 1).to(device)
        self.jacobian_y = torch.zeros(self.Nelementy-1, 1).to(device)
        
        self.init_elem()
        
    def init_elem(self):    
        # Compute quadrature nodes in physical elements
        self.x_quad_elem = self.gridx[:-2, 0:1] + (self.gridx[2:, 0:1] - self.gridx[:-2, 0:1]) / 2 * (self.x_quad.view(1, -1) + 1)
        self.y_quad_elem = self.gridy[:-2, 0:1] + (self.gridy[2:, 0:1] - self.gridy[:-2, 0:1]) / 2 * (self.y_quad.view(1, -1)+ 1)
        
        
        self.jacobian_x  = ((self.gridx[2:, 0:1] - self.gridx[:-2, 0:1]) / 2)
        self.jacobian_y  = ((self.gridy[2:, 0:1] - self.gridy[:-2, 0:1]) / 2)
        self.jacobian    = self.jacobian_x * self.jacobian_y.view(1, -1)
        
        nx = np.arange(1, self.Nelementx)
        ny = np.arange(1, self.Nelementy)
        Nx, Ny = np.meshgrid(nx, ny)
        Nx, Ny = Nx.flatten()[:, None], Ny.flatten()[:, None]
        
        self.node_list = np.hstack((Nx, Ny))
        self.N_nodes = self.node_list.shape[0]
        
    
        self.testx_quad_elem   = self.test_func(self.x_quad)
        self.d1testx_quad_elem = self.d1test_func(self.x_quad)
        
        self.testy_quad_elem   = self.test_func(self.y_quad)
        self.d1testy_quad_elem = self.d1test_func(self.y_quad)
        
        self.inv_gram = self.inverse_gram_mat().to(device)
        
        # Precompute constants for vectorized loss
        self.Nex = self.Nelementx - 1
        self.Ney = self.Nelementy - 1
        self.Nq  = self.x_quad.shape[0]
        # Quadrature weights
        wqx = self.w_quad[:, 0]
        wqy = self.w_quad[:, 1]
        self.wqx_q = wqx.view(1, 1, 1, -1)
        self.wqy_q = wqy.view(1, 1, 1, -1)
        # Jacobian expansions
        self.jac_exp  = self.jacobian.view(self.Nex, self.Ney, 1, 1, 1)
        self.jacx_exp = self.jacobian_x.view(self.Nex, 1, 1, 1, 1)
        self.jacy_exp = self.jacobian_y.view(1, self.Ney, 1, 1, 1)
        # Test‐function expansions
        self.testx_e   = self.testx_quad_elem.view(1, 1, 1, self.Nq)
        self.d1testx_e = self.d1testx_quad_elem.view(1, 1, 1, self.Nq)
        self.testy_e   = self.testy_quad_elem.view(1, 1, 1, self.Nq)
        self.d1testy_e = self.d1testy_quad_elem.view(1, 1, 1, self.Nq)
        # Precompute coefficient fields a(x,y) and f(x,y) on element×quad grid
        XQ = self.x_quad_elem.view(self.Nex, 1, self.Nq)
        YQ = self.y_quad_elem.view(1, self.Ney, self.Nq)
        flat_xy = torch.stack((
            XQ.expand(self.Nex, self.Ney, self.Nq).reshape(-1),
            YQ.expand(self.Nex, self.Ney, self.Nq).reshape(-1)
        ), dim=1)
        raw_xy = flat_xy.detach().cpu().numpy().astype(np.float32)
        a_np = self.a_func(raw_xy).astype(np.float32)
        f_raw = self.f_func(flat_xy[:,0:1], flat_xy[:,1:2]).detach().cpu().numpy()
        f_np  = f_raw.astype(np.float32)
        self.a_elem = torch.from_numpy(a_np).view(self.Nex, self.Ney, 1, 1, self.Nq).to(device)
        self.f_elem = torch.from_numpy(f_np).view(self.Nex, self.Ney, 1, 1, self.Nq).to(device)

    def forward(self, x, y):
        a = torch.cat((x, y), dim=-1)   
        output = self.linears[0](a)
        output = self.activation(output) 
        for i in range(1, len(self.layers)-2):
            z = self.linears[i](output)
            output = self.activation(z)   
        output = self.linears[-1](output)
        return output
    
    def inverse_gram_mat(self):
        """Build and invert Gram matrix for H1 seminorm residual coupling."""
        gram_mat = torch.zeros(self.N_nodes, self.N_nodes)
        
        for i in range(self.N_nodes):
            for j in range(self.N_nodes):
                if abs(self.node_list[i, 0] - self.node_list[j, 0]) == 1 and abs(self.node_list[i, 1] - self.node_list[j, 1]) == 1:
                    gram_mat[i, j] = self.dx/6 * self.dy/6 - 1/self.dx * self.dy/6 - 1/self.dy * self.dx/6
                elif (self.node_list[i, 0] - self.node_list[j, 0]) == 0 and abs(self.node_list[i, 1] - self.node_list[j, 1]) == 1:
                    gram_mat[i, j] = 2*self.dx/3 * self.dy/6 + 2/self.dx * self.dy/6 - 2*self.dx/3 * 1/self.dy
                elif abs(self.node_list[i, 0] - self.node_list[j, 0]) == 1 and (self.node_list[i, 1] - self.node_list[j, 1]) == 0:
                    gram_mat[i, j] = self.dx/6 * 2*self.dy/3 - 1/self.dx * 2*self.dy/3 + self.dx/6 * 2/self.dy
                elif (self.node_list[i, 0] - self.node_list[j, 0]) == 0 and (self.node_list[i, 1] - self.node_list[j, 1]) == 0:
                    gram_mat[i, j] = 2*self.dx/3 * 2*self.dy/3 + 2/self.dx * 2*self.dy/3 + 2*self.dx/3 * 2/self.dy
        
        inv_gram = torch.linalg.inv(gram_mat)
        return inv_gram.float()
    
    def test_func(self, x_np):
        """Linear hat: φ(x) = 1±x on [-1,1]."""
        x_np = x_np.detach().cpu().numpy().flatten()
        testf_total = np.zeros(x_np.shape[0])
        testf_total = (1 + x_np)
        arg = np.argwhere(x_np > 0.0)
        testf_total[arg] = (1 - x_np[arg])
        testf_total = torch.from_numpy(testf_total.astype(np.float32)).to(device)
        return testf_total
    
    def d1test_func(self, x_np):
        """Derivative of hat: φ′(x) = ∓1."""
        x_np = x_np.detach().cpu().numpy().flatten()
        d1testf_total = np.zeros(x_np.shape[0])
        argp = np.argwhere(x_np > 0.0)
        argn = np.argwhere(x_np < 0.0)
        d1testf_total[argp] = - 1.0
        d1testf_total[argn] = 1.0
        
        d1testf_total = torch.from_numpy(d1testf_total.astype(np.float32)).to(device)
        return d1testf_total  
       
    def loss(self):
        # Vectorized residual + boundary loss
        # Build global quadrature point arrays
        XQ = self.x_quad_elem.view(self.Nex, 1, self.Nq)
        YQ = self.y_quad_elem.view(1, self.Ney, self.Nq)
        X_flat = XQ.expand(self.Nex, self.Ney, self.Nq).reshape(-1,1).requires_grad_()
        Y_flat = YQ.expand(self.Nex, self.Ney, self.Nq).reshape(-1,1).requires_grad_()

        # Forward pass at all quad points
        u_all = self.forward(X_flat, Y_flat)
        # Compute grads w.r.t. inputs
        u_x_all = autograd.grad(u_all, X_flat, torch.ones_like(u_all), create_graph=True)[0]
        u_y_all = autograd.grad(u_all, Y_flat, torch.ones_like(u_all), create_graph=True)[0]
        # Reshape to element-quadrature grid
        ux = u_x_all.view(self.Nex, self.Ney, 1, 1, self.Nq)
        uy = u_y_all.view(self.Nex, self.Ney, 1, 1, self.Nq)

        # Compute variational integrand a ∇u · ∇v + f·v
        t1 = (self.jac_exp/self.jacx_exp) * self.a_elem * ux * self.wqx_q * self.wqy_q * self.d1testx_e * self.testy_e
        t2 = (self.jac_exp/self.jacy_exp) * self.a_elem * uy * self.wqx_q * self.wqy_q * self.testx_e * self.d1testy_e
        t3 =  self.jac_exp           * self.f_elem * self.wqx_q * self.wqy_q * self.testx_e * self.testy_e
        integrand = t1 + t2 - t3

        # Sum over quadrature points
        u_elem = integrand.sum(dim=-1).view(-1, 1)
        # L_r = R^T G^{-1} R
        lossr = (u_elem.T @ (self.inv_gram @ u_elem)).squeeze()

        # Boundary mean‐square loss
        ub = [
            self.forward(self.xy_BC_bottom[:,0:1], self.xy_BC_bottom[:,1:2]),
            self.forward(self.xy_BC_top   [:,0:1], self.xy_BC_top   [:,1:2]),
            self.forward(self.xy_BC_left  [:,0:1], self.xy_BC_left  [:,1:2]),
            self.forward(self.xy_BC_right [:,0:1], self.xy_BC_right [:,1:2]),
        ]
        lossb = sum(u.square().mean() for u in ub)

        return lossr, lossb
