import numpy as np
import torch
import os
from matplotlib import pyplot as plt

import time
from GaussJacobiQuadRule_V3 import Jacobi, DJacobi, GaussLobattoJacobiWeights
from scipy.interpolate import RegularGridInterpolator
from D_data import create_residual_data, create_BC_data
from D_rebano_activation import P

from D_rebano import ReBaNO
from D_rebano_train import rebano_train
from D_rebano_precomp import autograd_calculations

plt.rc('font', size=10)

torch.set_default_dtype(torch.float)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Current Device: {device}")

# activation functions in PINNs 
class custom_act(torch.nn.Module):
    def __init__(self):
        super(custom_act, self).__init__()
    
    def forward(self, x):
        return torch.sin(x)
    
def f_ext(x, y):
    return torch.ones(x.shape[0], 1)



n_train = 1000
n_test  = 200

# Domain and collocation points data
Xi, Xf = 0.0, 1.0
Yi, Yf = Xi, Xf
Nc, N_test = 100, 100
N_BC = 100
N_quad = 16

Nex, Ney = 8, 8
Ntest_func = (Nex-1) * (Ney-1)

residual_data = create_residual_data(Xi, Xf, Yi, Yf, Nc, N_test)
xy_resid      = residual_data[0].to(device)
xy_test       = residual_data[1].to(device) 

boundary_data = create_BC_data(Xi, Xf, Yi, Yf, N_BC)
BC_xy_bottom  = boundary_data[0].to(device)
BC_xy_top     = boundary_data[1].to(device)
BC_xy_left    = boundary_data[2].to(device)
BC_xy_right   = boundary_data[3].to(device)

[X_quad, WX_quad] = GaussLobattoJacobiWeights(N_quad, 0, 0)
Y_quad, WY_quad = (X_quad, WX_quad)

xx, yy = np.meshgrid(X_quad, Y_quad)
wxx, wyy = np.meshgrid(WX_quad, WY_quad)
XY_quad_train = np.hstack((xx.flatten()[:, None], yy.flatten()[:, None]))
WXY_quad_train = np.hstack((wxx.flatten()[:, None], wyy.flatten()[:, None]))

XY_quad_train  = torch.from_numpy(XY_quad_train.astype(np.float32)).to(device)
WXY_quad_train = torch.from_numpy(WXY_quad_train.astype(np.float32)).to(device)

# Load training and test data
K = 8
Nx = 100
path = "../data/"
# rf_coef = np.load(path + fr"Random_NS_rf_coef_{K}_s{Nx}.npy")
a_func = np.load(path + fr'darcy_input_a_K_{K}_s{Nx}.npy')
u_sol  = np.load(path + fr"darcy_output_u_K_{K}_s{Nx}.npy")

test_on_ood = True

a_train = a_func[:, :, :n_train]
u_train = u_sol[:, :, :n_train]

if test_on_ood:
    a_func_ood = np.load(path + fr'darcy_input_a_K_{K}_s{Nx}_ood.npy')
    u_sol_ood  = np.load(path + fr"darcy_output_u_K_{K}_s{Nx}_ood.npy")
    a_test   = a_func_ood[:, :, :n_test]
    u_test   = u_sol_ood[:, :, :n_test]
    datapath = "../data/ood/speedup/"
else:
    a_test  = a_func[:, :, n_train:n_train+n_test]
    u_test  = u_sol[:, :, n_train:n_train+n_test]
    datapath = "../data/speedup/"

grid_x = torch.linspace(Xi, Xf, Nex+1).view(-1, 1).to(device)
grid_y = torch.linspace(Yi, Yf, Ney+1).view(-1, 1).to(device)

x_quad_elem = grid_x[:-2, 0:1] + (grid_x[2:, 0:1] - grid_x[:-2, 0:1]) / 2 * (XY_quad_train[:, 0] + 1)
y_quad_elem = grid_y[:-2, 0:1] + (grid_y[2:, 0:1] - grid_y[:-2, 0:1]) / 2 * (XY_quad_train[:, 1] + 1)

XQ = x_quad_elem[:,None,:].expand(Nex-1, Ney-1, XY_quad_train.shape[0])
YQ = y_quad_elem[None,:,:].expand(Nex-1, Ney-1, XY_quad_train.shape[0])

XY_flat = torch.stack((XQ.reshape(-1), YQ.reshape(-1)), dim=1)
X_flat  = XQ.reshape(-1, 1).clone().requires_grad_()
Y_flat  = YQ.reshape(-1, 1).clone().requires_grad_()

xgrid = np.linspace(Xi, Xf, Nx+1)
ygrid = np.linspace(Yi, Yf, Nx+1)

number_of_neurons = 48

# customized activation functions 
P_list = np.ones(number_of_neurons, dtype=object)

P_resid_values = torch.ones((xy_test.shape[0], number_of_neurons)).to(device)
Px_quad_element = torch.ones((Nex-1, Ney-1, XY_quad_train.shape[0], number_of_neurons)).to(device)
Py_quad_element = torch.ones((Nex-1, Ney-1, XY_quad_train.shape[0], number_of_neurons)).to(device)

P_BC_bottom = torch.ones((BC_xy_bottom.shape[0], number_of_neurons)).to(device)
P_BC_top    = torch.ones((BC_xy_top.shape[0], number_of_neurons)).to(device)
P_BC_left   = torch.ones((BC_xy_left.shape[0], number_of_neurons)).to(device)
P_BC_right  = torch.ones((BC_xy_right.shape[0], number_of_neurons)).to(device)

BC_x_bottom = BC_xy_bottom[:, 0:1]
BC_y_bottom = BC_xy_bottom[:, 1:2]
BC_x_top    = BC_xy_top[:, 0:1]
BC_y_top    = BC_xy_top[:, 1:2]
BC_x_left   = BC_xy_left[:, 0:1]
BC_y_left   = BC_xy_left[:, 1:2]
BC_x_right  = BC_xy_right[:, 0:1]
BC_y_right  = BC_xy_right[:, 1:2]

x_test = xy_test[:, 0:1].to(device)
y_test = xy_test[:, 1:2].to(device)

rebano_sols_train = np.ones((N_test+1, N_test+1, n_train), dtype = np.float32)
rebano_sols_test  = np.ones((N_test+1, N_test+1, n_test), dtype = np.float32)



layers_pinn = np.array([2, 40, 40, 40, 40, 40, 40, 1])
layers_rebano  = np.array([2, number_of_neurons, 1])

lr_rebano          = 0.8
epochs_rebano      = 100
epochs_rebano_test = 100


# precompute activation function and their derivatives
for i in range(number_of_neurons):
    path = fr"../data/Full-PINN-Data (Darcy) (K={K})/({i+1})"
    weights = []
    bias = []
    for k in range(len(layers_pinn)-1):
        weights.append(np.loadtxt(fr"{path}/saved_w{k+1}.txt"))
        bias.append(np.loadtxt(fr"{path}/saved_b{k+1}.txt"))
    P_list[i] = P(layers_pinn, weights, bias, custom_act()).to(device)
    
    P_resid_values[:, i][:, None] = P_list[i](x_test, y_test).detach()
    Px, Py = autograd_calculations(Nex, Ney, XY_quad_train.shape[0], X_flat, Y_flat, P_list[i])
    Px_quad_element[:, :, :, i] = Px.to(device) 
    Py_quad_element[:, :, :, i] = Py.to(device)
    
    P_BC_bottom[:, i][:, None] = P_list[i](BC_x_bottom, BC_y_bottom).detach().to(device)
    P_BC_top[:, i][:, None]    = P_list[i](BC_x_top, BC_y_top).detach().to(device)
    P_BC_left[:, i][:, None]   = P_list[i](BC_x_left, BC_y_left).detach().to(device)
    P_BC_right[:, i][:, None]  = P_list[i](BC_x_right, BC_y_right).detach().to(device)
    
    
    
    
losses_train = np.zeros(n_train, dtype = np.float32)
losses_test = np.zeros(n_test, dtype = np.float32)

validation = False

if validation:
    total_train_time_1 = time.perf_counter()
    for i in range(n_train):
        c_initial = torch.full((1, number_of_neurons), 1/number_of_neurons)
        
        a = RegularGridInterpolator(
            (xgrid, ygrid),
            a_train[:, :, i],
            method='cubic'
        )
        
        u_rebano  = u_train[:, :, i]
        
        # ReBaNO_NN = TReBaNO(layers_rebano, nu, P_list[0:i+1], c_initial, u0_rebano_train, f_hat).to(device)

        ReBaNO_NN = ReBaNO(layers_rebano, P_list, c_initial, a, f_ext, Nex, Ney, Ntest_func, XY_quad_train, WXY_quad_train, grid_x, grid_y, BC_xy_bottom, BC_xy_top, BC_xy_left, BC_xy_right, Px_quad_element, Py_quad_element, P_BC_bottom, P_BC_top, P_BC_left, P_BC_right).to(device)
        rebano_losses = rebano_train(ReBaNO_NN, epochs_rebano_test, lr_rebano, testing=True)
                
        losses_train[i] = rebano_losses
        
        u_rebano_pred = ReBaNO_NN(x_test, y_test)
        rebano_sols_train[:, :, i] = u_rebano_pred.detach().cpu().numpy().reshape(N_test+1, N_test+1).T
        
    total_train_time_2 = time.perf_counter()

    print(f'avg training time on training dataset : {(total_train_time_2 - total_train_time_1)/n_train} seconds')
    np.save(datapath + f'ReBaNO_solutions_train_K_{K}_n_{n_train}_s{Nx}.npy', rebano_sols_train)
    
    rel_err_rebano_train = np.zeros(n_train)
    for i in range(n_train):
        rel_err_rebano_train[i] = np.linalg.norm(rebano_sols_train[:, :, i] - u_train[::(Nx//N_test), ::(Nx//N_test), i])/np.linalg.norm(u_train[::(Nx//N_test), ::(Nx//N_test), i])
    
    np.save(datapath + f"ReBaNO_rel_err_train_K_{K}_n_{n_train}_s{Nx}.npy", rel_err_rebano_train)
    
    print(f'mean loss on training dataset: {np.mean(losses_train)}')
    print(f'mean rel train err: {np.mean(rel_err_rebano_train)}, largest rel train err: {np.max(rel_err_rebano_train)}')
    

total_test_time_1 = time.perf_counter()
for i in range(n_test):
    c_initial = torch.full((1, number_of_neurons), 1/number_of_neurons)
    
    a = RegularGridInterpolator(
        (xgrid, ygrid),
        a_test[:, :, i],
        method='cubic'
    )
    
    u_rebano  = u_test[:, :, i]
    
    # ReBaNO_NN = TReBaNO(layers_rebano, nu, P_list[0:i+1], c_initial, u0_rebano_train, f_hat).to(device)

    ReBaNO_NN = ReBaNO(layers_rebano, P_list, c_initial, a, f_ext, Nex, Ney, Ntest_func, XY_quad_train, WXY_quad_train, grid_x, grid_y, BC_xy_bottom, BC_xy_top, BC_xy_left, BC_xy_right, Px_quad_element, Py_quad_element, P_BC_bottom, P_BC_top, P_BC_left, P_BC_right).to(device)
    rebano_losses = rebano_train(ReBaNO_NN, epochs_rebano_test, lr_rebano, testing=True)
            
    losses_test[i] = rebano_losses
    
    u_rebano_pred = ReBaNO_NN(x_test, y_test)
    rebano_sols_test[:, :, i] = u_rebano_pred.detach().cpu().numpy().reshape(N_test+1, N_test+1).T
total_test_time_2 = time.perf_counter()

print(f'total training time on test dataset : {(total_test_time_2 - total_test_time_1)/n_test} seconds')
np.save(datapath + f'ReBaNO_solutions_test_K_{K}_n_{n_test}_s{Nx}.npy', rebano_sols_test)

rel_err_rebano_test  = np.zeros(n_test)

for i in range(n_test):
    rel_err_rebano_test[i] = np.linalg.norm(rebano_sols_test[:, :, i] - u_test[::(Nx//N_test), ::(Nx//N_test), i])/np.linalg.norm(u_test[::(Nx//N_test), ::(Nx//N_test), i])
    
np.save(datapath + f"ReBaNO_rel_err_test_K_{K}_n_{n_test}_s{Nx}.npy", rel_err_rebano_test)
    
print(f'mean loss on test dataset: {np.mean(losses_test)}')
print(f'mean rel test err: {np.mean(rel_err_rebano_test)}, largest rel test err: {np.max(rel_err_rebano_test)}')
