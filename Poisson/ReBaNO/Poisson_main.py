# Import and GPU Support
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import torch
import os
import sys
import time

from Poisson_data import create_residual_data, create_BC_data

# Full PINN
from Poisson_pinn import PINN
from Poisson_pinn_train import pinn_train

# ReBaNO
from Poisson_rebano_activation import P
from Poisson_rebano_precomp import autograd_calculations
from Poisson_rebano import ReBaNO
from Poisson_rebano_train import rebano_train

np.random.seed(0)
torch.manual_seed(0)

torch.set_default_dtype(torch.float)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Current Device: {device}')

# activation function in PINNs
class custom_act(torch.nn.Module):
    """Full PINN Activation Function"""
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.tanh(x)

n_train = 1000
n_test  = 200

# Domain and collocation points
Xi, Xf         =  0.0, 1.0
Nc, N_test     =  128, 128

x_grid = np.linspace(Xi, Xf, Nc+1)
x_grid = x_grid[0:-1]

residual_data  = create_residual_data(Xi, Xf, Nc, N_test)
x_resid        = residual_data[0].to(device)
x_test         = residual_data[1].to(device)

BC_data = create_BC_data(Xi, Xf)
BC_x    = BC_data[0].to(device)
BC_u    = BC_data[1].to(device)    

K = 10
Nx = 128

f_hat   = np.load(f"../data/poisson1d_input_f_K_{K}_s{Nx}.npy") 
u_exact = np.load(f"../data/poisson1d_output_u_K_{K}_s{Nx}.npy")

# Load trainind data

f_hat_train = torch.from_numpy(f_hat[:, :n_train].astype(np.float32)).to(device)
u_train     = torch.from_numpy(u_exact[:, :n_train].astype(np.float32)).to(device)

f_hat_test = torch.from_numpy(f_hat[:, n_train:(n_train+n_test)].astype(np.float32)).to(device)
u_test     = torch.from_numpy(u_exact[:, n_train:(n_train+n_test)].astype(np.float32)).to(device)

train_final_rebano   = True
number_of_neurons = 8
loss_list         = np.ones(number_of_neurons) # Store largest losses
ind_list          = np.ones(number_of_neurons, dtype=np.int32)
print(f"Expected Final ReBaNO Depth: {[1,number_of_neurons,1]}\n")

###############################################################################
#################################### Setup ####################################
###############################################################################

P_resid_values = torch.ones((x_resid.shape[0], number_of_neurons)).to(device)
P_BC_values    = torch.ones((   BC_x.shape[0], number_of_neurons)).to(device)
P_xx_term      = torch.ones((x_resid.shape[0], number_of_neurons)).to(device)

P_list = np.ones(number_of_neurons, dtype=object)

# training hyper-params

lr_pinn     = 0.0005
epochs_pinn = 400
tol_pinn    = 1e-6

layers_pinn = np.array([1, 20, 20, 20, 1])

lr_rebano          = 0.8
epochs_rebano      = 100
epochs_rebano_test = 100
test_cases      = n_test

# Save Data/Plot Options
save_data         = False
plot_pinn_loss    = False
plot_pinn_sol     = False
plot_largest_loss = False

pinn_train_times = np.ones(number_of_neurons)
rebano_train_times  = np.ones(number_of_neurons)

arg_max = 0
num_pre_neurons = 0 # number of pre-trained neurons

total_train_time_1 = time.perf_counter()
###############################################################################
################################ Training Loop ################################
###############################################################################

for i in range(0, number_of_neurons):
    if i < num_pre_neurons:
         # Load pre-trained neurons
        if i == 0:
            print("Loading pre-trained neurons")
            print("# of pre-trained neurons : ", num_pre_neurons)
        
        path = fr"../data/Full-PINN-Data (Poisson) (K={K})/({i+1})"
        weights = []
        bias = []
        for k in range(len(layers_pinn)-1):
            weights.append(np.loadtxt(fr"{path}/saved_w{k+1}.txt"))
            bias.append(np.loadtxt(fr"{path}/saved_b{k+1}.txt"))
        P_list[i] = P(layers_pinn, weights, bias, custom_act()).to(device)
        P_resid_values[:, i][:, None] = P_list[i](x_resid).detach()
        
        largest_param = np.loadtxt(fr"{path}/largest_param.txt")
        
        largest_case = largest_param[0]
        largest_loss = largest_param[1]
        
        ind_list[i]  = largest_case
        loss_list[i] = largest_loss
        
        arg_max = int(largest_case)
        
        layers_rebano = np.array([1, i+1, 1])
        
        P_xx       = autograd_calculations(x_resid, P_list[i])  
        P_xx_term[:,i][:,None] = P_xx
        
        P_BC_values[:,i][:,None]    = P_list[i](BC_x).detach()
        
    else:    
        print("******************************************************************")
        ########################### Full PINN Training ############################
        f_hat_pinn = f_hat_train[::(Nx//Nc), arg_max].view(-1,1)
        
        pinn = PINN(layers_pinn, f_hat_pinn, custom_act()).to(device)
        # print(f_hat.size(), BC_x.size(), x_resid.size())
        if (i+1 == number_of_neurons):
            print(f"Begin Final Full PINN Training: case = {arg_max} (Obtaining Neuron {i+1})")
        else:
            print(f"Begin Full PINN Training: case = {arg_max} (Obtaining Neuron {i+1})")
        
        pinn_train_time_1 = time.perf_counter()
        pinn_losses = pinn_train(pinn, x_resid, BC_x, BC_u, epochs_pinn, lr_pinn, tol_pinn)
        pinn_train_time_2 = time.perf_counter()
        print(f"PINN Training Time: {(pinn_train_time_2-pinn_train_time_1)/3600} Hours")
        
        pinn_pred = pinn(x_resid).detach().cpu().flatten().numpy()
        u_pinn = u_train[::(Nx//Nc), arg_max].cpu().numpy()
        
        pinn_rel_err = np.linalg.norm(pinn_pred - u_pinn)/np.linalg.norm(u_pinn)
        
        print(f'pinn rel err: {pinn_rel_err}')
        
        if plot_pinn_sol:
            fig, ax = plt.subplots(ncols=2, figsize=(10,4))
            ax[0].plot(x_grid,  u_pinn, linewidth=1.33)
            ax[1].plot(x_grid, pinn_pred, linewidth=1.33)
            fig.suptitle(f'case index = {arg_max}, rel err = {pinn_rel_err:.5f}')
            fig.tight_layout()
            plt.savefig(f'../figs/pinn_sol_#{i+1}_K_{K}.pdf')
            # plt.show()
            plt.close()
            
        
        weights = []
        bias    = []
        
        for k in range(len(layers_pinn)-1):
            weights.append(pinn.linears[k].weight.detach().cpu())
            bias.append(pinn.linears[k].bias.detach().cpu())
        
        P_list[i] = P(layers_pinn, weights, bias, custom_act()).to(device)

        print(f"\nCurrent ReBaNO Depth: [1,{i+1},1]")
        
        if (save_data):        
            path = fr"../data/Full-PINN-Data (Poisson) (K={K})/({i+1})"
            
            if not os.path.exists(path):
                os.makedirs(path)
            for k in range(len(layers_pinn)-1):
                np.savetxt(fr"{path}/saved_w{k+1}.txt", weights[k].numpy())
                np.savetxt(fr"{path}/saved_b{k+1}.txt", bias[k].numpy())
        

        if (i == number_of_neurons-1) and (train_final_rebano == False):
            break
            
        ############################ ReBaNO Training ############################    
        layers_rebano = np.array([1, i+1, 1])
        
        # Precompute residual and BC values, and derivatives of PINN solutions
        P_xx       = autograd_calculations(x_resid, P_list[i])  
        P_xx_term[:,i][:,None] = P_xx
        
        P_BC_values[:,i][:,None]    = P_list[i](BC_x).detach()
        P_resid_values[:,i][:,None] = P_list[i](x_resid).detach()
        
        largest_case = 0
        largest_loss = 0
        
        if (i+1 == number_of_neurons):
            print(f"\nBegin Final ReBaNO Training (Finding Neuron {i+2} / Using {i+1} Neurons)")
        else:
            print(f"\nBegin ReBaNO Training (Finding Neuron {i+2} / Using {i+1} Neurons)")
            
        rebano_train_time_1 = time.perf_counter()
       
        for j in range(n_train):
            c_initial  = torch.full((1,i+1), 1./(i+1))
            f_hat_rebano = f_hat_train[::(Nx//Nc), j].view(-1,1)
                
            ReBaNO_NN = ReBaNO(layers_rebano, P_list[0:i+1], c_initial, BC_u, f_hat_rebano, 
                         P_resid_values[:,0:i+1], P_BC_values[:,0:i+1], P_xx_term[:,0:i+1]).to(device)
            rebano_losses = rebano_train(ReBaNO_NN, x_resid, BC_x, epochs_rebano, lr_rebano, 
                                   j, largest_loss, largest_case)
            largest_loss = rebano_losses[0]
            largest_case = rebano_losses[1]
            arg_max      = largest_case

        rebano_train_time_2 = time.perf_counter()
        
        print("ReBaNO Training Completed")
        print(f"ReBaNO Training Time ({i+1} Neurons): {(rebano_train_time_2-rebano_train_time_1)/3600} Hours")
        print(f"\nLargest Loss (Using {i+1} Neurons): {largest_loss}")
        print(f"Parameter Case: {largest_case}")
        loss_list[i] = largest_loss
        ind_list[i]  = arg_max
        
        largest_param = np.array([arg_max, loss_list[i]])
        
        if (save_data):        
            path = fr"../data/Full-PINN-Data (Poisson) (K={K})/({i+1})"
            
            if not os.path.exists(path):
                os.makedirs(path)
            
            np.savetxt(fr"{path}/largest_param.txt", largest_param)
total_train_time_2 = time.perf_counter()


###############################################################################
# Results of largest loss, input functions chosen, and times may vary based on
# the initialization of full PINN and the final loss of the full PINN
print("******************************************************************")
print("*** Full PINN and ReBaNO Training Complete ***")
print(f"total training time : {(total_train_time_2 - total_train_time_1)/3600} hours")
print(f"Case indices selected by ReBaNO Depth {[1, number_of_neurons, 1]}: {ind_list}")
for j in range(number_of_neurons-1):
    print(f"Largest Loss of ReBaNO Depth {[1,j+1,1]}: {loss_list[j]}")
if (train_final_rebano):
    print(f"Largest Loss of ReBaNO Depth {[1,j+2,1]}: {loss_list[-1]}")
        
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
    plt.savefig("./ReBaNO-largest-losses-{}-{}.png".format(number_of_neurons, n_train))
    plt.close()
