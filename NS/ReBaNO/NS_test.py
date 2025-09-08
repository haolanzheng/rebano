import numpy as np
import torch
import os
from matplotlib import pyplot as plt

import time

from NS_data import create_residual_data, create_ICBC_data
from NS_rebano_activation import P
from NS_rebano_precomp import autograd_calculations, Pt_nu_lap_vor
from NS_rebano import ReBaNO
from NS_rebano_train import rebano_train

plt.rc('font', size=10)

torch.set_default_dtype(torch.float)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Current Device: {device}")

# activation function in PINNs
class custom_act(torch.nn.Module):
    def __init__(self):
        super(custom_act, self).__init__()
    
    def forward(self, x):
        return torch.cos(x)

n_train = 1000
n_test  = 200

nu = 0.025

# Domain and collocation points data
Xi, Xf = 0.0, 2*np.pi
Yi, Yf = Xi, Xf
Ti, Tf = 0.0, 10.0
Nc, N_test     = 100, 100
BC_pts, IC_pts = 100, 100
Nt = 100

residual_data = create_residual_data(Xi, Xf, Yi, Yf, Ti, Tf, Nc, Nt, N_test)
xt_resid      = residual_data[0].to(device)
xt_test       = residual_data[1].to(device) 

ICBC_data = create_ICBC_data(Xi, Xf, Yi, Yf, Ti, Tf, Nt, BC_pts, IC_pts)
IC_xt     = ICBC_data[0].to(device)
BC_xt_bottom = ICBC_data[1].to(device)
BC_xt_top    = ICBC_data[2].to(device)
BC_xt_left   = ICBC_data[3].to(device)
BC_xt_right  = ICBC_data[4].to(device)

# Load training and test data
K = 8
Nx = 100
path = "../data/"
curl_f = np.load(path + fr'ns_input_curl_f_K_{K}_s{Nx}.npy').transpose(1, 0, 2)
omega0 = np.load(path + fr"ns_omega0_K_{K}_s{Nx}.npy").T
omega  = np.load(path + fr"ns_output_omega_K_{K}_s{Nx}.npy").transpose(1, 0, 2)

curl_f_ood = np.load(path + fr'nx_input_curl_f_K_{K}_s{Nx}_ood.npy').transpose(1, 0, 2)
omega_ood  = np.load(path + fr"ns_output_omega_K_{K}_s{Nx}_ood.npy").transpose(1, 0, 2)

omega0 = torch.from_numpy(omega0.reshape(-1,1).astype(np.float32)).to(device)

test_on_ood = False

curl_f_train = torch.from_numpy(curl_f[:, :, :n_train].astype(np.float32)).to(device)
omega_train = torch.from_numpy(omega[:, :, :n_train]).to(device)

if test_on_ood:
    curl_f_test = torch.from_numpy(curl_f_ood[:, :, :n_test].astype(np.float32)).to(device)
    omega_test = torch.from_numpy(omega_ood[:, :, :n_test]).to(device)
    datapath = "../data/ood/"
else:
    curl_f_test = torch.from_numpy(curl_f[:, :, n_train:n_train+n_test].astype(np.float32)).to(device)
    omega_test = torch.from_numpy(omega[:, :, n_train:n_train+n_test]).to(device)
    datapath = "../data/"

# omega_train = omega[:, :, :n_train]
# omega_test  = omega[:, :, n_train:(n_train+n_test)]

number_of_neurons = 20

P_list = np.ones(number_of_neurons, dtype=object)

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

x_test = xt_test[:, 0:1].to(device)
y_test = xt_test[:, 1:2].to(device)
t_test = xt_test[:, 2:3].to(device)

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

rebano_sols_train = np.ones((N_test, N_test, n_train), dtype = np.float32)
rebano_sols_test  = np.ones((N_test, N_test, n_test), dtype = np.float32)



layers_pinn = np.array([3, 20, 20, 20, 20, 20, 20, 2])
layers_rebano  = np.array([3, number_of_neurons, 1])

lr_rebano          = 0.005
epochs_rebano      = 5000
epochs_rebano_test = 5000



for i in range(number_of_neurons):
    path = fr"../data/Full-PINN-Data (NS) (K={K})/({i+1})"
    weights = []
    bias = []
    for k in range(len(layers_pinn)-1):
        weights.append(np.loadtxt(fr"{path}/saved_w{k+1}.txt"))
        bias.append(np.loadtxt(fr"{path}/saved_b{k+1}.txt"))
    P_list[i] = P(layers_pinn, weights, bias, custom_act()).to(device)
    
    P_t, P_x, P_xx, P_y, P_yy, vel, lap_psi = autograd_calculations(x_test, y_test, t_test, P_list[i]) 
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
    
    P_resid_vorstr_values[:, :, i] = P_list[i](x_test, y_test, t_test).detach()
    P_vel[:, :, i] = vel
    P_lap_psi[:, i][:, None] = lap_psi
    
    
    
    
    
losses_train = np.zeros(n_train, dtype = np.float32)
losses_test = np.zeros(n_test, dtype = np.float32)


validation = True

if validation:
    total_train_time_1 = time.perf_counter()
    for i in range(n_train):
        c_initial = torch.full((1, number_of_neurons), 1/number_of_neurons)
        curl_f_rebano_train = curl_f_train[::(Nx//Nc), ::(Nx//Nc), i].reshape(-1, 1)
        curl_f_rebano_train = curl_f_rebano_train.repeat(Nt + 1, 1).to(device)
        omega_rebano  = omega_train[::(Nx//Nc), ::(Nx//Nc), i].reshape(-1, 1).to(device)
        ReBaNO_NN = ReBaNO(layers_rebano, nu, P_list, c_initial, omega0, curl_f_rebano_train, omega_rebano,
                    P_resid_vorstr_values, P_IC_values, P_BC_bottom, 
                    P_BC_top, P_BC_left, P_BC_right, Pt_nu_lap_omega, P_vel, P_x_term, P_y_term, P_lap_psi).to(device)
        rebano_losses = rebano_train(ReBaNO_NN, nu, xt_test, IC_xt, BC_xt_bottom, BC_xt_top, BC_xt_left,        
                            BC_xt_right, epochs_rebano, lr_rebano, testing=True)
                
        losses_train[i] = rebano_losses
        
        u_rebano_pred = ReBaNO_NN(testing=True, test_x=x_test, test_y=y_test, test_t=t_test)[(-N_test**2):, 0]
        rebano_sols_train[:, :, i] = u_rebano_pred.detach().cpu().numpy().reshape(N_test, N_test).T
        
    total_train_time_2 = time.perf_counter()
    
    print(f'total training time on training dataset : {total_train_time_2 - total_train_time_1} seconds')
    np.save(datapath + f'ReBaNO_solutions_train_K_{K}_n_{n_train}_s{N_test}.npy', rebano_sols_train)
    
    print(f'Mean train loss: {np.mean(losses_train)}')
          
    omega_train = omega_train.permute(1, 0, 2).cpu().numpy()
    rel_err_rebano_train = np.zeros(n_train)
    
    for i in range(n_train):
        rel_err_rebano_train[i] = np.linalg.norm(rebano_sols_train[:, :, i] - omega_train[::(Nx//N_test), ::(Nx//N_test), i])/np.linalg.norm(omega_train[::(Nx//N_test), ::(Nx//N_test), i])
    
    np.save(datapath + f"ReBaNO_rel_err_train_K_{K}_n_{n_train}_s{N_test}.npy", rel_err_rebano_train)
    print(f'mean rel train err: {np.mean(rel_err_rebano_train)}, largest rel train err: {np.max(rel_err_rebano_train)}')

total_test_time_1 = time.perf_counter()
for i in range(n_test):
    c_initial = torch.full((1, number_of_neurons), 1/number_of_neurons)
    curl_f_rebano_test = curl_f_test[::(Nx//Nc), ::(Nx//Nc), i].reshape(-1, 1)
    curl_f_rebano_test = curl_f_rebano_test.repeat(Nt + 1, 1).to(device)
    omega_rebano  = omega_test[::(Nx//Nc), ::(Nx//Nc), i].reshape(-1, 1).to(device)
    ReBaNO_NN = ReBaNO(layers_rebano, nu, P_list, c_initial, omega0, curl_f_rebano_test, omega_rebano,
                 P_resid_vorstr_values, P_IC_values, P_BC_bottom, 
                 P_BC_top, P_BC_left, P_BC_right, Pt_nu_lap_omega, P_vel, P_x_term, P_y_term, P_lap_psi).to(device)
    rebano_losses = rebano_train(ReBaNO_NN, nu, xt_test, IC_xt, BC_xt_bottom, BC_xt_top, BC_xt_left,        
                           BC_xt_right, epochs_rebano, lr_rebano, testing=True)
    
    losses_test[i] = rebano_losses
    
    u_rebano_pred = ReBaNO_NN(testing=True, test_x=x_test, test_y=y_test, test_t=t_test)[(-N_test**2):, 0]
    rebano_sols_test[:, :, i] = u_rebano_pred.detach().cpu().numpy().reshape(N_test, N_test).T
total_test_time_2 = time.perf_counter()

print(f'total training time on test dataset : {total_test_time_2 - total_test_time_1} seconds')
print(f'Mean test loss: {np.mean(losses_test)}')

np.save(datapath + f'ReBaNO_solutions_test_K_{K}_n_{n_test}_s{N_test}.npy', rebano_sols_test)

"""
path = '../data/'
rebano_sols_train = np.load(path + fr'ReBaNO_solutions_train_K_{K}.npy')
rebano_sols_test  = np.load(path + fr'ReBaNO_solutions_test_K_{K}.npy')
print(rebano_sols_train.shape, rebano_sols_test.shape)
"""

omega_test = omega_test.permute(1, 0, 2).cpu().numpy()

rel_err_rebano_test  = np.zeros(n_test)

for i in range(n_test):
    rel_err_rebano_test[i] = np.linalg.norm(rebano_sols_test[:, :, i] - omega_test[::(Nx//N_test), ::(Nx/N_test), i])/np.linalg.norm(omega_test[::(Nx//N_test), ::(Nx//N_test), i])
    

np.save(datapath + f"ReBaNO_rel_err_test_K_{K}_n_{n_test}_s{N_test}.npy", rel_err_rebano_test)

print(f'mean rel test err: {np.mean(rel_err_rebano_test)}, largest rel test err: {np.max(rel_err_rebano_test)}')


"""

xgrid = np.linspace(0, 2*np.pi, Nc+1)
xgrid = xgrid[0:-1]
X, Y = np.meshgrid(xgrid, xgrid)

arg_max_train = np.argmax(rel_err_rebano_train)
arg_max_test  = np.argmax(rel_err_rebano_test)

fig, ax = plt.subplots(2, ncols=3, figsize=(12, 8), sharey=True, sharex=True)
im1 = ax[0, 0].pcolormesh(X, Y, omega[:, :, arg_max_train], shading='gouraud', cmap='rainbow')
im2 = ax[0, 1].pcolormesh(X, Y, rebano_sols_train[:, :, arg_max_train], shading='gouraud', cmap='rainbow')
im3 = ax[0, 2].pcolormesh(X, Y, abs(omega[:, :, arg_max_train] - rebano_sols_train[:, :, arg_max_train]), shading='gouraud', cmap='rainbow')
im4 = ax[1, 0].pcolormesh(X, Y, omega[:, :, n_train+arg_max_test], shading='gouraud', cmap='rainbow')
im5 = ax[1, 1].pcolormesh(X, Y, rebano_sols_test[:, :, arg_max_test], shading='gouraud', cmap='rainbow')
im6 = ax[1, 2].pcolormesh(X, Y, abs(omega[:, :, n_train+arg_max_test] - rebano_sols_test[:, :, arg_max_test]), shading='gouraud', cmap='rainbow')
fig.colorbar(im1, ax=ax[0, 0])
fig.colorbar(im2, ax=ax[0, 1])
fig.colorbar(im3, ax=ax[0, 2])
fig.colorbar(im4, ax=ax[1, 0])
fig.colorbar(im5, ax=ax[1, 1])
fig.colorbar(im6, ax=ax[1, 2])
fig.suptitle(f"max rel train err: {np.max(rel_err_rebano_train):.4f}, max rel test err: {np.max(rel_err_rebano_test):.4f}")
fig.tight_layout()
plt.savefig("../figs/ReBaNO_largest_rel_err.pdf")
"""