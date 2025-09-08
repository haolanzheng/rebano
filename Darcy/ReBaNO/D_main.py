# Import and GPU Support
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import sys
import time
import random

from GaussJacobiQuadRule_V3 import Jacobi, DJacobi, GaussLobattoJacobiWeights
from scipy.interpolate import RegularGridInterpolator
from D_data import create_residual_data, create_BC_data

# Full PINN
from D_vpinn import VPINN
from D_vpinn_train import vpinn_train

# ReBaNO
from D_rebano_activation import P
from D_rebano import ReBaNO
from D_rebano_train import rebano_train
from D_rebano_precomp import autograd_calculations


torch.set_default_dtype(torch.float)
torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Current Device: {device}")

# activation function in PINNs
class custom_act(torch.nn.Module):
    def __init__(self):
        super(custom_act, self).__init__()
    
    def forward(self, x):
        return torch.sin(x)
    
def f_ext(x, y):
    return torch.ones(x.shape[0], 1)


n_train = 1000

# Domain and collocation points data
Xi, Xf = 0.0, 1.0
Yi, Yf = Xi, Xf
Nc, N_test = 100, 100
N_BC = 100

N_quad = 20
Nex, Ney = 8, 8
Ntest_func = (Nex - 1) * (Ney - 1)

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

# Load training data
K = 8
Nx = 100
path = "../data/"

a_func = np.load(path + fr'darcy_input_a_K_{K}_s{Nx}.npy')
u_sol  = np.load(path + fr"darcy_output_u_K_{K}_s{Nx}.npy")

a_train = a_func[:, :, :n_train]
u_train = u_sol[:, :, :n_train]

x = torch.linspace(Xi, Xf, Nc+1)

train_final_rebano   = True
number_of_neurons = 48
loss_list         = np.ones(number_of_neurons)
ind_list          = np.ones(number_of_neurons, dtype=np.int32)
print(f"Expected Final ReBaNO Depth: {[2,number_of_neurons,1]}\n")

###############################################################################
#################################### Setup ####################################
###############################################################################

P_list = np.ones(number_of_neurons, dtype=object)

# training hyper-perams
lr_adam    = 0.005
lr_lbfgs = 0.8
epochs_adam  = 600
epochs_lbfgs = 2000

layers_pinn = np.array([2, 40, 40, 40, 40, 40, 40, 1])
tol_adam    = 1e-7

lr_rebano          = 0.8
epochs_rebano      = 100
epochs_rebano_test = 100
test_cases         = 200

# Save Data/Plot Options
save_data         = False
plot_pinn_loss    = False
plot_pinn_sol     = False
plot_largest_loss = False

#generate mesh to find U0-pred for the whole domain

P_resid_values = torch.ones((xy_resid.shape[0], number_of_neurons)).to(device)
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

grid_x = torch.linspace(Xi, Xf, Nex+1).view(-1,1).to(device)
grid_y = torch.linspace(Yi, Yf, Ney+1).view(-1,1).to(device)

x_quad_elem = grid_x[:-2, 0:1] + (grid_x[2:, 0:1] - grid_x[:-2, 0:1]) / 2 * (XY_quad_train[:, 0] + 1)
y_quad_elem = grid_y[:-2, 0:1] + (grid_y[2:, 0:1] - grid_y[:-2, 0:1]) / 2 * (XY_quad_train[:, 1] + 1)

XQ = x_quad_elem[:,None,:].expand(Nex-1, Ney-1, XY_quad_train.shape[0])
YQ = y_quad_elem[None,:,:].expand(Nex-1, Ney-1, XY_quad_train.shape[0])

XY_flat = torch.stack((XQ.reshape(-1), YQ.reshape(-1)), dim=1)
X_flat  = XQ.reshape(-1, 1).clone().requires_grad_()
Y_flat  = YQ.reshape(-1, 1).clone().requires_grad_()

xgrid = np.linspace(Xi, Xf, Nx+1)
ygrid = np.linspace(Yi, Yf, Nx+1)


X, Y = np.meshgrid(xgrid, ygrid)

x_resid = xy_resid[:, 0:1].to(device)
y_resid = xy_resid[:, 1:2].to(device)

pinn_train_times = np.ones(number_of_neurons)
rebano_train_times  = np.ones(number_of_neurons)
arg_max = 0 # random.randint(0, n_train-1)
num_pre_neurons = 0
total_train_time_1 = time.perf_counter()
###############################################################################
################################ Training Loop ################################
###############################################################################
for i in range(number_of_neurons):
    
    ########################### Full PINN Training ############################
    
    if i < num_pre_neurons:
        # Load pre-trained neurons
        if i == 0:
            print("Loading pre-trained neurons")
            print("# of pre-trained neurons : ", num_pre_neurons)
        
        path = fr"../data/Full-PINN-Data (Darcy) (K={K})/({i+1})"
        weights = []
        bias = []
        for k in range(len(layers_pinn)-1):
            weights.append(np.loadtxt(fr"{path}/saved_w{k+1}.txt"))
            bias.append(np.loadtxt(fr"{path}/saved_b{k+1}.txt"))
        P_list[i] = P(layers_pinn, weights, bias, custom_act()).to(device)
        P_resid_values[:, i][:, None] = P_list[i](x_resid, y_resid).detach()
        Px, Py = autograd_calculations(Nex, Ney, XY_quad_train.shape[0], X_flat, Y_flat, P_list[i])
        Px_quad_element[:, :, :, i] = Px.to(device) 
        Py_quad_element[:, :, :, i] = Py.to(device)
        
        P_BC_bottom[:, i][:, None] = P_list[i](BC_x_bottom, BC_y_bottom).detach().to(device)
        P_BC_top[:, i][:, None]    = P_list[i](BC_x_top, BC_y_top).detach().to(device)
        P_BC_left[:, i][:, None]   = P_list[i](BC_x_left, BC_y_left).detach().to(device)
        P_BC_right[:, i][:, None]  = P_list[i](BC_x_right, BC_y_right).detach().to(device)
        
        largest_param = np.loadtxt(fr"{path}/largest_param.txt")
        
        largest_case = largest_param[0]
        largest_loss = largest_param[1]
        
        ind_list[i]  = largest_case
        loss_list[i] = largest_loss
        
        arg_max = int(largest_case)
        
        layers_rebano = np.array([2, i+1, 1])
        
    else:
        print("******************************************************************")
       
        a = RegularGridInterpolator(
            (xgrid, ygrid),
            a_train[:, :, arg_max],
            method='cubic'
        )
        
        pinn_train_time1 = time.perf_counter()
        # vpinn = VPINN(layers_pinn, custom_act(), a, f_ext, Nex, Ney, Ntest_func, XY_quad_train, WXY_quad_train, grid_x, grid_y).to(device)
        vpinn = VPINN(layers_pinn, custom_act(), a, f_ext, Nex, Ney, Ntest_func, XY_quad_train, WXY_quad_train, grid_x, grid_y, BC_xy_bottom, BC_xy_top, BC_xy_left, BC_xy_right).to(device)
        
        if (i+1 == number_of_neurons):
            print(f"Begin Final Full PINN Training: case={arg_max} (Obtaining Neuron {i+1})")
        else:
            print(f"Begin Full PINN Training: case = {arg_max} (Obtaining Neuron {i+1})")
        
        pinn_losses = vpinn_train(vpinn, lr_adam, epochs_adam, tol_adam, lr_lbfgs, epochs_lbfgs)
        
        pinn_train_time2 = time.perf_counter()
        
        print(f"PINN Training Time: {(pinn_train_time2-pinn_train_time1)/3600} Hours")
        
        pinn_pred = vpinn(x_resid, y_resid).detach().cpu().numpy()
        pinn_pred = pinn_pred.reshape(Nc+1, Nc+1).T
        u_pinn = u_train[::(Nx//Nc), ::(Nx//Nc), arg_max]
        
        pinn_rel_err = np.linalg.norm(pinn_pred - u_pinn)/np.linalg.norm(u_pinn)
        
        print(f'pinn rel err: {pinn_rel_err}')
        if plot_pinn_sol:
            fig, ax = plt.subplots(ncols=2, figsize=(10,4))
            im1 = ax[0].pcolormesh(X, Y, u_pinn, shading='gouraud', cmap='rainbow')
            im2 = ax[1].pcolormesh(X, Y, pinn_pred, shading='gouraud', cmap='rainbow')
            fig.colorbar(im1, ax=ax[0])
            fig.colorbar(im2, ax=ax[1])
            fig.suptitle(f'case index = {arg_max}, rel err = {pinn_rel_err:.4f}')
            fig.tight_layout()
            plt.savefig(f'../figs/K={K}/pinn_sol_#{i+1}.pdf')
            plt.close()
        
        
        weights = []
        bias    = []
        
        for k in range(len(layers_pinn)-1):
            weights.append(vpinn.linears[k].weight.detach().cpu())
            bias.append(vpinn.linears[k].bias.detach().cpu())
        
        P_list[i] = P(layers_pinn, weights, bias, custom_act()).to(device)

        print(f"\nCurrent ReBaNO Depth: [2,{i+1},1]")
        
        if (save_data):        
            path = fr"../data/Full-PINN-Data (Darcy) (K={K})/({i+1})"
            
            if not os.path.exists(path):
                os.makedirs(path)
            for k in range(len(layers_pinn)-1):
                np.savetxt(fr"{path}/saved_w{k+1}.txt", weights[k].numpy())
                np.savetxt(fr"{path}/saved_b{k+1}.txt", bias[k].numpy())
      
        
        
    
        if (i == number_of_neurons-1) and (train_final_rebano == False):
            break

        ############################ ReBaNO Training ############################
        layers_rebano = np.array([2, i+1, 1])
        
        # Precompute all values for rebano training
        P_resid_values[:, i][:, None] = P_list[i](x_resid, y_resid).detach()
        Px, Py = autograd_calculations(Nex, Ney, XY_quad_train.shape[0], X_flat, Y_flat, P_list[i])
        Px_quad_element[:, :, :, i] = Px.to(device) 
        Py_quad_element[:, :, :, i] = Py.to(device)
        
        P_BC_bottom[:, i][:, None] = P_list[i](BC_x_bottom, BC_y_bottom).detach().to(device)
        P_BC_top[:, i][:, None]    = P_list[i](BC_x_top, BC_y_top).detach().to(device)
        P_BC_left[:, i][:, None]   = P_list[i](BC_x_left, BC_y_left).detach().to(device)
        P_BC_right[:, i][:, None]  = P_list[i](BC_x_right, BC_y_right).detach().to(device)
        # Finding The Next Neuron
        largest_case = 0
        largest_loss = 0

        if (i+1 == number_of_neurons):
            print("\nBegin Final ReBaNO Training (Largest Loss Training)")
        else:
            print(f"\nBegin ReBaNO Training (Finding Neuron {i+2} / Largest Loss Training)")

        rebano_train_time_1 = time.perf_counter()
        for j in range(n_train):

            c_initial = torch.full((1, i+1), 1/(i+1))
    
            a = RegularGridInterpolator(
                (xgrid, ygrid),
                a_train[:, :, j],
                method='cubic'
            )
            
            u_rebano  = u_train[:, :, j]
            
            # ReBaNO_NN = TReBaNO(layers_rebano, nu, P_list[0:i+1], c_initial, u0_rebano_train, f_hat).to(device)

            ReBaNO_NN = ReBaNO(layers_rebano, P_list[:(i+1)], c_initial, a, f_ext, Nex, Ney, Ntest_func, XY_quad_train, WXY_quad_train, grid_x, grid_y, BC_xy_bottom, BC_xy_top, BC_xy_left, BC_xy_right, Px_quad_element[:, :, :, 0:(i+1)], Py_quad_element[:, :, :, 0:(i+1)], P_BC_bottom[:, 0:i+1], P_BC_top[:, 0:i+1], P_BC_left[:, 0:i+1], P_BC_right[:, 0:i+1]).to(device)
            rebano_losses = rebano_train(ReBaNO_NN, epochs_rebano, lr_rebano, j, largest_loss, largest_case)
            
            largest_loss = rebano_losses[0]
            largest_case = rebano_losses[1]
            arg_max = largest_case
        rebano_train_time_2 = time.perf_counter()
        print("ReBaNO Training Completed")
        print(f"\nReBaNO Training Time ({i+1} Neurons): {(rebano_train_time_2-rebano_train_time_1)/3600} Hours")
        
        loss_list[i] = largest_loss
        ind_list[i]  = int(arg_max)
            
        print(f"\nLargest Loss (Using {i+1} Neurons): {largest_loss}")
        print(f"Parameter Case: {largest_case}")

        largest_param = np.array([arg_max, loss_list[i]])
        
        if (save_data):        
            path = fr"../data/Full-PINN-Data (Darcy) (K={K})/({i+1})"
            
            if not os.path.exists(path):
                os.makedirs(path)
            
            np.savetxt(fr"{path}/largest_param.txt", largest_param)
total_train_time_2 = time.perf_counter()                       

###############################################################################
# Results of largest loss, parameters chosen, and times may vary based on
# the initialization of full PINN and the final loss of the full PINN
print("******************************************************************")
print("*** Full PINN and ReBaNO Training Complete ***")
print(f"Total Training Time: {(total_train_time_2-total_train_time_1)/3600} Hours\n")
print(f"Final ReBaNO Depth: {[2,len(P_list),1]}")
# print(f"\nActivation Function Parameters: \n{nu_neurons}\n")

print(f"Case indices selected by ReBaNO Depth {[2, number_of_neurons, 1]}: {ind_list}")
for j in range(number_of_neurons-1):
    print(f"Largest Loss of ReBaNO Depth {[2,j+1,1]}: {loss_list[j]}")
if (train_final_rebano):
    print(f"Largest Loss of ReBaNO Depth {[2,j+2,1]}: {loss_list[-1]}")
        
if (plot_largest_loss):
    plt.figure(dpi=150, figsize=(10,8))
    
    if (train_final_rebano):
        range_end = number_of_neurons + 1
        list_end  = number_of_neurons
    else:
        range_end = number_of_neurons 
        list_end  = number_of_neurons - 1
        
    plt.plot(range(1,range_end), loss_list[:list_end], marker='o', markersize=7, 
             c="k", linewidth=3)
    
    plt.grid(True)
    plt.xlim(1,max(range(1,range_end)))
    plt.xticks(range(1,range_end))
    
    plt.yscale("log") 
    plt.xlabel("Number of Neurons",      fontsize=17.5)
    plt.ylabel("Largest Loss",           fontsize=17.5)
    plt.title("ReBaNO Largest Losses", fontsize=17.5)
    plt.show()
