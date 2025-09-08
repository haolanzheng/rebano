# Import and GPU Support
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import sys
import time
import random


from NS_data import create_residual_data, create_ICBC_data

# Full PINN
from NS_pinn import PINN
from NS_pinn_train import pinn_train


# Burgers ReBaNO
from NS_rebano_activation import P
from NS_rebano_precomp import autograd_calculations, Pt_nu_lap_vor
from NS_rebano import ReBaNO
from NS_rebano_train import rebano_train


torch.set_default_dtype(torch.float)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Current Device: {device}")

class custom_act(torch.nn.Module):
    def __init__(self):
        super(custom_act, self).__init__()
    
    def forward(self, x):
        return torch.cos(x)

torch.manual_seed(0)
np.random.seed(0)

n_train = 1000

nu = 0.025

# Domain and collocation pints data
Xi, Xf = 0.0, 2*np.pi
Yi, Yf = Xi, Xf
Ti, Tf = 0.0, 10.0
Nc, N_test     = 100, 100
Nt = 128
BC_pts, IC_pts = 100, 100

residual_data = create_residual_data(Xi, Xf, Yi, Yf, Ti, Tf, Nc, Nt, N_test)
xt_resid      = residual_data[0].to(device)
xt_test       = residual_data[1].to(device) 

ICBC_data = create_ICBC_data(Xi, Xf, Yi, Yf, Ti, Tf, Nt, BC_pts, IC_pts)
IC_xt     = ICBC_data[0].to(device)
BC_xt_bottom = ICBC_data[1].to(device)
BC_xt_top    = ICBC_data[2].to(device)
BC_xt_left   = ICBC_data[3].to(device)
BC_xt_right  = ICBC_data[4].to(device)

# Load training data
K = 8
Nx = 100
path = "../data/"

curl_f = np.load(path + fr'ns_input_curl_f_K_{K}_s{Nx}.npy').transpose(1, 0, 2)
omega0 = np.load(path + fr"ns_omega0_K_{K}_s{Nx}.npy").T
omega  = np.load(path + fr"ns_output_omega_K_{K}_s{Nx}.npy").transpose(1, 0, 2)

omega0 = torch.from_numpy(omega0.reshape(-1,1).astype(np.float32)).to(device)
curl_f_train = torch.from_numpy(curl_f[:, :, :n_train].astype(np.float32)).to(device)
omega_train  = torch.from_numpy(omega[:, :, :n_train].astype(np.float32)).to(device)
"""
n = 20
coarse_idx  = [(Nx//n)*i for i in range(n)]
omega_train_coarse = omega_train[coarse_idx, :, :]
omega_train_coarse = omega_train_coarse[:, coarse_idx, :]

x = torch.linspace(Xi, Xf, Nc+1)
x = x[0:-1]
t = torch.linspace(Ti, Tf, Nt+1)

x_data, y_data = x[coarse_idx], x[coarse_idx]
X_data, Y_data = torch.meshgrid(x_data, y_data, indexing='xy')

x_data, y_data = X_data.flatten()[:, None], Y_data.flatten()[:, None]
t_data = t[coarse_idx]
xy_data = torch.hstack((x_data, y_data))
t_data  = t_data[:, None].repeat(1, xy_data.shape[0])
xy_data = xy_data.repeat(t_data.shape[0], 1)
t_data  = t_data.flatten()[:, None]
xt_data = torch.hstack((xy_data, t_data)).to(device)
"""

train_final_rebano   = True
number_of_neurons = 20
loss_list         = np.ones(number_of_neurons)
ind_list          = np.ones(number_of_neurons, dtype=np.int32)
print(f"Expected Final ReBaNO Depth: {[3,number_of_neurons,2]}\n")

###############################################################################
#################################### Setup ####################################
###############################################################################

P_list = np.ones(number_of_neurons, dtype=object)

lr_adam    = 0.005
lr_weights = 0.005
lr_lbfgs = 0.8
epochs_pinn  = 60000
epochs_lbfgs = 2000

layers_pinn = np.array([3, 20, 20, 20, 20, 20, 20, 2])
tol_adam    = 5e-8

lr_rebano          = 0.005
epochs_rebano      = 5000
epochs_rebano_test = 5000
test_cases      = 200

# Save Data/Plot Options
save_data         = False
plot_pinn_loss    = False
plot_pinn_sol     = False
plot_largest_loss = False

#generate mesh to find U0-pred for the whole domain

P_resid_vorstr_values = torch.ones((xt_resid.shape[0], 2, number_of_neurons)).to(device)
P_IC_values    = torch.ones((   IC_xt.shape[0], number_of_neurons)).to(device)
P_BC_bottom    = torch.ones((   BC_xt_bottom.shape[0], 2, number_of_neurons)).to(device)
P_BC_top       = torch.ones((   BC_xt_bottom.shape[0], 2, number_of_neurons)).to(device)
P_BC_left      = torch.ones((   BC_xt_bottom.shape[0], 2, number_of_neurons)).to(device)
P_BC_right     = torch.ones((   BC_xt_bottom.shape[0], 2, number_of_neurons)).to(device)

P_t_term  = torch.ones((xt_resid.shape[0], number_of_neurons)).to(device)
P_x_term  = torch.ones((xt_resid.shape[0], number_of_neurons)).to(device)
P_xx_term = torch.ones((xt_resid.shape[0], number_of_neurons)).to(device)
P_y_term  = torch.ones((xt_resid.shape[0], number_of_neurons)).to(device)
P_yy_term = torch.ones((xt_resid.shape[0], number_of_neurons)).to(device)
P_vel     = torch.ones((xt_resid.shape[0], 2, number_of_neurons)).to(device)
P_lap_psi = torch.ones((xt_resid.shape[0], number_of_neurons)).to(device)
Pt_nu_lap_omega = torch.ones((xt_resid.shape[0], number_of_neurons)).to(device)

xgrid = np.linspace(Xi, Xf, Nc+1)
ygrid = np.linspace(Yi, Yf, Nc+1)
xgrid, ygrid = xgrid[:-1], ygrid[:-1]
X, Y = np.meshgrid(xgrid, ygrid)

x_resid = xt_resid[:, 0:1].to(device)
y_resid = xt_resid[:, 1:2].to(device)
t_resid = xt_resid[:, 2:3].to(device)

IC_x = IC_xt[:, 0:1].to(device)
IC_y = IC_xt[:, 1:2].to(device)
IC_t = IC_xt[:, 2:3].to(device)
BC_x_top = BC_xt_top[:, 0:1].to(device)
BC_y_top = BC_xt_top[:, 1:2].to(device)
BC_t_top = BC_xt_top[:, 2:3].to(device)
BC_x_bottom = BC_xt_bottom[:, 0:1].to(device)
BC_y_bottom = BC_xt_bottom[:, 1:2].to(device)
BC_t_bottom = BC_xt_bottom[:, 2:3].to(device)
BC_x_left = BC_xt_left[:, 0:1].to(device)
BC_y_left = BC_xt_left[:, 1:2].to(device)
BC_t_left = BC_xt_left[:, 2:3].to(device)
BC_x_right = BC_xt_right[:, 0:1].to(device)
BC_y_right = BC_xt_right[:, 1:2].to(device)
BC_t_right = BC_xt_right[:, 2:3].to(device)

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
        """Load pretrained neurons"""
        if i == 0:
            print("Loading pre-trained neurons")
            print("# of pre-trained neurons : ", num_pre_neurons)
        
        path = fr"../data/Full-PINN-Data (NS) (K={K})/({i+1})"
        weights = []
        bias = []
        for k in range(len(layers_pinn)-1):
            weights.append(np.loadtxt(fr"{path}/saved_w{k+1}.txt"))
            bias.append(np.loadtxt(fr"{path}/saved_b{k+1}.txt"))
        P_list[i] = P(layers_pinn, weights, bias, custom_act()).to(device)
        P_resid_vorstr_values[:, :, i] = P_list[i](x_resid, y_resid, t_resid).detach()
        
        largest_param = np.loadtxt(fr"{path}/largest_param.txt")
        
        largest_case = largest_param[0]
        largest_loss = largest_param[1]
        
        ind_list[i]  = largest_case
        loss_list[i] = largest_loss
        
        arg_max = int(largest_case)
        
        layers_rebano = np.array([3, i+1, 1])
        
        P_t, P_x, P_xx, P_y, P_yy, vel, lap_psi = autograd_calculations(x_resid, y_resid, t_resid, P_list[i]) 
        Pt_nu_lap_omega[:, i][:, None] = Pt_nu_lap_vor(nu, P_t, P_xx, P_yy)

        P_t_term[:,i][:,None]  = P_t # torch.from_numpy(u_t)
        P_x_term[:,i][:,None]  = P_x # torch.from_numpy(u_x)
        P_xx_term[:,i][:,None] = P_xx # torch.from_numpy(u_xx)
        P_y_term[:,i][:,None]  = P_y 
        P_yy_term[:,i][:,None] = P_yy
        
        
        P_IC_values[:, i][:,None] = P_list[i](IC_x, IC_y, IC_t)[:, 0:1].detach() 
        P_BC_top[:, :, i]    = P_list[i](BC_x_top, BC_y_top, BC_t_top).detach() 
        P_BC_bottom[:, :, i] = P_list[i](BC_x_bottom, BC_y_bottom, BC_t_bottom).detach()
        P_BC_left[:, :, i]   = P_list[i](BC_x_left, BC_y_left, BC_t_left).detach()
        P_BC_right[:, :, i]  = P_list[i](BC_x_right, BC_y_right, BC_t_right).detach()
        
        P_resid_vorstr_values[:, :, i] = P_list[i](x_resid, y_resid, t_resid).detach()
        P_vel[:, :, i] = vel
        P_lap_psi[:, i][:, None] = lap_psi
        
    else:
        print("******************************************************************")
        curl_f_pinn = curl_f_train[::(Nx//Nc), ::(Nx//Nc), arg_max].reshape(-1, 1).to(device)
        curl_f_pinn = curl_f_pinn.repeat(Nt+1, 1)
        # omega_pinn  = omega_train_coarse[:, :, arg_max].reshape(-1, 1).to(device)
        
        pinn = PINN(layers_pinn, nu, curl_f_pinn, omega0, custom_act()).to(device)
        
        pinn_train_time_1 = time.perf_counter()
        
        if (i+1 == number_of_neurons):
            print(f"Begin Final Full PINN Training: case={arg_max} (Obtaining Neuron {i+1})")
        else:
            print(f"Begin Full PINN Training: case = {arg_max} (Obtaining Neuron {i+1})")
        
        pinn_losses = pinn_train(pinn, nu, xt_resid, IC_xt, BC_xt_bottom, BC_xt_top, BC_xt_left, 
                                 BC_xt_right, epochs_pinn, lr_adam, tol_adam, lr_lbfgs, epochs_lbfgs,
                                 using_data=False)

        pinn_train_time_2 = time.perf_counter()
        print(f"PINN Training Time: {(pinn_train_time_2-pinn_train_time_1)/3600} Hours")
        
        pinn_pred = pinn(x_resid, y_resid, t_resid).detach().cpu().numpy()
        pinn_pred = pinn_pred[(-Nc**2):, 0].reshape(Nc, Nc).T
        omega_pinn = omega_train[::(Nx//Nc), ::(Nx//Nc), arg_max].cpu().numpy().T
        
        pinn_rel_err = np.linalg.norm(pinn_pred - omega_pinn)/np.linalg.norm(omega_pinn)
        
        print(f'pinn rel err: {pinn_rel_err}')
        if plot_pinn_sol:
            fig, ax = plt.subplots(ncols=2, figsize=(10,4))
            im1 = ax[0].pcolormesh(X, Y, omega_pinn, shading='gouraud', cmap='rainbow')
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
            weights.append(pinn.linears[k].weight.detach().cpu())
            bias.append(pinn.linears[k].bias.detach().cpu())
        
        P_list[i] = P(layers_pinn, weights, bias, custom_act()).to(device)

        print(f"\nCurrent ReBaNO Depth: [3,{i+1},1]")
        
        if (save_data):        
            path = fr"../data/Full-PINN-Data (NS) (K={K})/({i+1})"
            
            if not os.path.exists(path):
                os.makedirs(path)
            for k in range(len(layers_pinn)-1):
                np.savetxt(fr"{path}/saved_w{k+1}.txt", weights[k].numpy())
                np.savetxt(fr"{path}/saved_b{k+1}.txt", bias[k].numpy())
      
        
    
        if (i == number_of_neurons-1) and (train_final_rebano == False):
            break

        ############################ ReBaNO Training ############################
        layers_rebano = np.array([3, i+1, 1])
        
        # Precompute all values for rebano training
        P_t, P_x, P_xx, P_y, P_yy, vel, lap_psi = autograd_calculations(x_resid, y_resid, t_resid, P_list[i]) 
        Pt_nu_lap_omega[:, i][:, None] = Pt_nu_lap_vor(nu, P_t, P_xx, P_yy)

        P_t_term[:,i][:,None]  = P_t 
        P_x_term[:,i][:,None]  = P_x 
        P_xx_term[:,i][:,None] = P_xx 
        P_y_term[:,i][:,None]  = P_y 
        P_yy_term[:,i][:,None] = P_yy
        
        
        P_IC_values[:, i][:,None] = P_list[i](IC_x, IC_y, IC_t)[:, 0:1].detach() 
        P_BC_top[:, :, i]    = P_list[i](BC_x_top, BC_y_top, BC_t_top).detach() 
        P_BC_bottom[:, :, i] = P_list[i](BC_x_bottom, BC_y_bottom, BC_t_bottom).detach()
        P_BC_left[:, :, i]   = P_list[i](BC_x_left, BC_y_left, BC_t_left).detach()
        P_BC_right[:, :, i]  = P_list[i](BC_x_right, BC_y_right, BC_t_right).detach()
        
        P_resid_vorstr_values[:, :, i] = P_list[i](x_resid, y_resid, t_resid).detach()
        P_vel[:, :, i] = vel
        P_lap_psi[:, i][:, None] = lap_psi
        
        
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
            curl_f_rebano = curl_f_train[::(Nx//Nc), ::(Nx//Nc), j].reshape(-1, 1).to(device)
            curl_f_rebano = curl_f_rebano.repeat(Nt+1, 1)
            omega_rebano  = omega_train[::(Nx//Nc), ::(Nx//Nc), j].reshape(-1, 1).to(device)
            
            P_resid_rebano = P_resid_vorstr_values[:, :, :i+1]
            P_IC_rebano    = P_IC_values[:, :i+1]
            P_BC_bottom_rebano = P_BC_bottom[:, :, :i+1]
            P_BC_top_rebano    = P_BC_top[:, :, :i+1]
            P_BC_left_rebano  = P_BC_left[:, :, :i+1]
            P_BC_right_rebano = P_BC_right[:, :, :i+1]
            P_vel_rebano = P_vel[:, :, :i+1]
            P_x_rebano   = P_x_term[:, :i+1]
            P_y_rebano   = P_y_term[:, :i+1]
            P_lap_psi_rebano = P_lap_psi[:, :i+1]
            Pt_nu_lap_omega_rebano = Pt_nu_lap_omega[:, :i+1]

            ReBaNO_NN = ReBaNO(layers_rebano, nu, P_list[0:i+1], c_initial, omega0, curl_f_rebano, omega_rebano, 
                            P_resid_rebano, P_IC_rebano, P_BC_bottom_rebano, P_BC_top_rebano, P_BC_left_rebano, P_BC_right_rebano, Pt_nu_lap_omega_rebano, P_vel_rebano, P_x_rebano, P_y_rebano, P_lap_psi_rebano).to(device)
            rebano_losses = rebano_train(ReBaNO_NN, nu, xt_resid, IC_xt, BC_xt_bottom, 
                                         BC_xt_top, BC_xt_left, BC_xt_right, epochs_rebano, lr_rebano, j, largest_loss, largest_case)
            
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
            path = fr"../data/Full-PINN-Data (NS) (K={K})/({i+1})"
            
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

print(f"Case indices selected by ReBaNO Depth {[3, number_of_neurons, 1]}: {ind_list}")
for j in range(number_of_neurons-1):
    print(f"Largest Loss of ReBaNO Depth {[2,j+1,2]}: {loss_list[j]}")
if (train_final_rebano):
    print(f"Largest Loss of ReBaNO Depth {[2,j+2,2]}: {loss_list[-1]}")
        
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
