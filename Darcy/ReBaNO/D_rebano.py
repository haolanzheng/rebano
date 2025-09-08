import torch
from torch import nn
import numpy as np
from GaussJacobiQuadRule_V3 import Jacobi, DJacobi

# Select GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float)
torch.manual_seed(0)

class ReBaNO(nn.Module):
    """ReBaNO of 2d Darcy flow problem"""
    def __init__(self,
                 layers,
                 P_list,        # activation function list
                 initial_c,     # initial c values
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
                 Px_quad_elem,  # u_x values on quad nodes in all elements
                 Py_quad_elem,  # u_y values on quad nodes in all elements
                 P_BC_bottom,
                 P_BC_top,
                 P_BC_left,
                 P_BC_right):
        super(ReBaNO, self).__init__()
        self.to(device)

        # Network architecture
        self.layers     = layers
        self.activation = P_list
        self.linears    = nn.ModuleList([nn.Linear(layers[1], layers[2], bias=False)])
        self.linears[-1].weight.data = initial_c

        # PDE data and quadrature
        self.a_func       = a_func
        self.f_func       = f_func
        self.Nelementx    = N_elementx
        self.Nelementy    = N_elementy
        self.Ntest_func   = Ntest_func
        self.x_quad       = X_quad[:, 0:1]
        self.y_quad       = X_quad[:, 1:2]
        self.w_quad       = W_quad
        self.gridx        = gridx
        self.gridy        = gridy
        self.dx           = gridx[1] - gridx[0]
        self.dy           = gridy[1] - gridy[0]

        # Boundary collocation points
        self.xy_BC_bottom = xy_BC_bottom
        self.xy_BC_top    = xy_BC_top
        self.xy_BC_left   = xy_BC_left
        self.xy_BC_right  = xy_BC_right

        # Element‐wise linear inputs
        self.Px_quad_element = Px_quad_elem
        self.Py_quad_element = Py_quad_elem
        self.P_BC_bottom     = P_BC_bottom
        self.P_BC_top        = P_BC_top
        self.P_BC_left       = P_BC_left
        self.P_BC_right      = P_BC_right

        # Prepare storage for element quadrature nodes & Jacobians
        self.x_quad_elem  = torch.zeros(N_elementx-1, self.x_quad.shape[0], requires_grad=True).to(device)
        self.y_quad_elem  = torch.zeros(N_elementy-1, self.y_quad.shape[0], requires_grad=True).to(device)
        self.jacobian_x   = torch.zeros(N_elementx-1, 1).to(device)
        self.jacobian_y   = torch.zeros(N_elementy-1, 1).to(device)

        # Build all precomputed tensors
        self.init_elem()

    def init_elem(self):
        # Compute quadrature nodes in physical elements
        self.x_quad_elem = self.gridx[:-2,0:1] + (self.gridx[2:,0:1] - self.gridx[:-2,0:1]) / 2 * (self.x_quad.view(1,-1) + 1)
        self.y_quad_elem = self.gridy[:-2,0:1] + (self.gridy[2:,0:1] - self.gridy[:-2,0:1]) / 2 * (self.y_quad.view(1,-1) + 1)
        # Jacobians
        self.jacobian_x = (self.gridx[2:,0:1] - self.gridx[:-2,0:1]) / 2
        self.jacobian_y = (self.gridy[2:,0:1] - self.gridy[:-2,0:1]) / 2
        self.jacobian   = self.jacobian_x * self.jacobian_y.view(1, -1)

        # Build inverse Gram matrix once
        self.inv_gram = self.inverse_gram_mat().to(device)

        # 1D test functions at quadrature nodes
        self.testx_quad_elem   = self.test_func(self.x_quad)
        self.d1testx_quad_elem = self.d1test_func(self.x_quad)
        self.testy_quad_elem   = self.test_func(self.y_quad)
        self.d1testy_quad_elem = self.d1test_func(self.y_quad)

        # Precompute all the tensors needed by `loss`
        Nex = self.Nelementx - 1
        Ney = self.Nelementy - 1
        Nq  = self.x_quad.shape[0]

        # Quadrature weights
        wqx = self.w_quad[:,0]
        wqy = self.w_quad[:,1]
        self.wqx_q = wqx.view(1,1,1,-1)
        self.wqy_q = wqy.view(1,1,1,-1)

        # Expand Jacobians for broadcast over elements & quadrature
        self.jac_exp  = self.jacobian.view(Nex, Ney, 1,1,1)
        self.jacx_exp = self.jacobian_x.view(Nex,1,1,1,1)
        self.jacy_exp = self.jacobian_y.view(1,Ney,1,1,1)

        # Expand test funcs
        self.testx_e   = self.testx_quad_elem.view(1,1,1,Nq)
        self.d1testx_e = self.d1testx_quad_elem.view(1,1,1,Nq)
        self.testy_e   = self.testy_quad_elem.view(1,1,1,Nq)
        self.d1testy_e = self.d1testy_quad_elem.view(1,1,1,Nq)

        # Precompute coefficient fields a(x,y) and f(x,y) on the full element×quad grid
        XQ = self.x_quad_elem.view(Nex,1,Nq)
        YQ = self.y_quad_elem.view(1,Ney,Nq)
        flat_xy = torch.stack((
            XQ.expand(Nex, Ney, Nq).reshape(-1),
            YQ.expand(Nex, Ney, Nq).reshape(-1)
        ), dim=1)
        a_np = self.a_func(flat_xy.detach().cpu().numpy().astype(np.float32))
        f_np = self.f_func(flat_xy[:,0:1], flat_xy[:,1:2])
        self.a_elem = torch.from_numpy(a_np.astype(np.float32)).view(Nex, Ney,1,1,Nq).to(device)
        self.f_elem = torch.from_numpy(f_np.detach().cpu().numpy().astype(np.float32)).view(Nex, Ney,1,1,Nq).to(device)

    def forward(self, test_x, test_y):
        # Assemble hidden representation at collocation points
        a = torch.zeros(test_x.shape[0], 0, device=device)
        for act in self.activation:
            a = torch.cat((a, act(test_x, test_y)), dim=1)
        return self.linears[-1](a)

    def inverse_gram_mat(self):
        """Build and invert Gram matrix for H1 seminorm residual coupling."""
        Nx = np.arange(1, self.Nelementx)
        Ny = np.arange(1, self.Nelementy)
        coords = np.vstack(np.meshgrid(Nx, Ny, indexing='ij')).reshape(2, -1).T
        Nn = coords.shape[0]
        G = torch.zeros(Nn, Nn, device=device)
        for i in range(Nn):
            for j in range(Nn):
                dx = abs(coords[i,0] - coords[j,0])
                dy = abs(coords[i,1] - coords[j,1])
                if dx==0 and dy==0:
                    G[i,j] = 2*self.dx/3 * 2*self.dy/3 + 2/self.dx * 2*self.dy/3 + 2*self.dx/3 * 2/self.dy
                elif dx==0 and dy==1:
                    G[i,j] = 2*self.dx/3 * self.dy/6 + 2/self.dx * self.dy/6 - 2*self.dx/3 * 1/self.dy
                elif dx==1 and dy==0:
                    G[i,j] = self.dx/6 * 2*self.dy/3 - 1/self.dx * 2*self.dy/3 + self.dx/6 * 2/self.dy
                elif dx==1 and dy==1:
                    G[i,j] = self.dx/6 * self.dy/6 - 1/self.dx * self.dy/6 - 1/self.dy * self.dx/6
        return torch.linalg.inv(G).float()

    def test_func(self, x):
        """Linear hat: φ(x) = 1±x on [-1,1]."""
        x_np = x.detach().cpu().numpy()
        phi = np.where(x_np>0, 1 - x_np, 1 + x_np)
        return torch.from_numpy(phi.astype(np.float32)).to(device)

    def d1test_func(self, x):
        """Derivative of hat: φ′(x) = ∓1."""
        x_np = x.detach().cpu().numpy()
        dphi = np.where(x_np>0, -1.0, 1.0)
        return torch.from_numpy(dphi.astype(np.float32)).to(device)

    def loss(self):
        """Vectorized residual + boundary loss"""
        Nex = self.Nelementx - 1
        Ney = self.Nelementy - 1
        Nq  = self.x_quad.shape[0]
        width = self.layers[1]

        # Evaluate interior solution gradients once
        Pxf = self.Px_quad_element.reshape(-1, width)
        Pyf = self.Py_quad_element.reshape(-1, width)
        ux_flat = self.linears[-1](Pxf).view(Nex, Ney,1,1,Nq)
        uy_flat = self.linears[-1](Pyf).view(Nex, Ney,1,1,Nq)

        # Compute residual integrand: a ∇u · ∇v + f·v
        t1 = (self.jac_exp/self.jacx_exp) * self.a_elem * ux_flat * self.wqx_q*self.wqy_q * self.d1testx_e * self.testy_e
        t2 = (self.jac_exp/self.jacy_exp) * self.a_elem * uy_flat * self.wqx_q*self.wqy_q * self.testx_e * self.d1testy_e
        t3 =  self.jac_exp           * self.f_elem * self.wqx_q*self.wqy_q * self.testx_e * self.testy_e
        integrand = t1 + t2 - t3

        # Sum over quadrature and project through inverse Gram
        u_e = integrand.sum(dim=-1).reshape(-1,1)               # (Nelem²×Ntest²,1)
        # print(u_e.dtype, self.inv_gram.dtype)
        lossr = (u_e.T @ (self.inv_gram @ u_e)).squeeze()       # scalar

        # Boundary mean‐square loss
        ub = [
            self.linears[-1](self.P_BC_bottom),
            self.linears[-1](self.P_BC_top),
            self.linears[-1](self.P_BC_left),
            self.linears[-1](self.P_BC_right),
        ]
        lossb = sum(u.square().mean() for u in ub)

        return lossr, lossb