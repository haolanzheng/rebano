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

class custom_act(torch.nn.Module):
    """Full PINN Activation Function"""
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.tanh(x)

n_train = 1000
n_test  = 1000

# Domain and collocation points data
Xi, Xf         = 0.0, 1.0

res = [32, 50, 64, 100, 200, 256, 400, 512, 800, 1024]

for N_test in res:
    print("Nx : ", N_test, "\n")
    x_grid = np.linspace(Xi, Xf, N_test+1)
    x_grid = x_grid[0:-1]

    residual_data  = create_residual_data(Xi, Xf, N_test, N_test)
    x_resid        = residual_data[0].to(device)
    x_test         = residual_data[1].to(device)

    BC_data = create_BC_data(Xi, Xf)
    BC_x    = BC_data[0].to(device)
    BC_u    = BC_data[1].to(device)    

    K = 10
    Nx = N_test

    f_hat   = np.load(f"../data/poisson1d_input_f_K_{K}_s{Nx}.npy") 
    u_exact = np.load(f"../data/poisson1d_output_u_K_{K}_s{Nx}.npy")


    # Loading training and test data
    test_on_ood = False
    f_hat_train = torch.from_numpy(f_hat[:, :n_train].astype(np.float32)).to(device)
    u_train     = u_exact[:, :n_train].astype(np.float32)

    if test_on_ood:
        f_hat_ood   = np.load(f"../data/poisson1d_input_f_K_{K}_s{Nx}_ood.npy") 
        u_exact_ood = np.load(f"../data/poisson1d_output_u_K_{K}_s{Nx}_ood.npy")
        # rf_coef_ood = np.load(f"../data/Random_Poisson1d_trunc_rf_coef_{K}_s{Nx}_ood.npy")
        f_hat_test = torch.from_numpy(f_hat_ood[:, :n_test].astype(np.float32)).to(device)
        u_test     = u_exact_ood[:, :n_test].astype(np.float32)
        datapath = "../data/ood/lbfgs/"
    else:
        f_hat_test = torch.from_numpy(f_hat[:, :n_test].astype(np.float32)).to(device)
        u_test     = u_exact[:, :n_test].astype(np.float32)
        datapath = "../data/lbfgs/"

    number_of_neurons = 8

    losses_train = np.ones(n_train)
    losses_test  = np.ones(n_test)

    rebano_sol_train = torch.ones(x_test.shape[0], n_train, dtype=torch.float).to(device)
    rebano_sol_test  = torch.ones(x_test.shape[0], n_test, dtype=torch.float).to(device)

    P_resid_values = torch.ones((x_test.shape[0], number_of_neurons)).to(device)
    P_BC_values    = torch.ones((   BC_x.shape[0], number_of_neurons)).to(device)
    P_xx_term      = torch.ones((x_test.shape[0], number_of_neurons)).to(device)

    P_list = np.ones(number_of_neurons, dtype=object)


    layers_pinn = np.array([1, 20, 20, 20, 1])
    layers_rebano  = np.array([1, number_of_neurons, 1])

    lr_rebano          = 0.8
    epochs_rebano      = 100
    epochs_rebano_test = 100

    # Loading pre-trained PINNs
    for i in range(number_of_neurons):
        path = fr"../data/Full-PINN-Data (Poisson) (K={K})/({i+1})"
        weights = []
        bias = []
        for k in range(len(layers_pinn)-1):
            weights.append(np.loadtxt(fr"{path}/saved_w{k+1}.txt"))
            bias.append(np.loadtxt(fr"{path}/saved_b{k+1}.txt"))
        P_list[i] = P(layers_pinn, weights, bias, custom_act()).to(device)
        
        P_resid_values[:, i][:, None] = P_list[i](x_test).detach()
        
        P_xx = autograd_calculations(x_test, P_list[i])  
        
        P_xx_term[:, i][:,None]   = P_xx  
        P_BC_values[:, i][:,None] = P_list[i](BC_x).detach()

    validation = False

    if validation:
        total_time = 0
        for i in range(n_train):
            torch.cuda.synchronize()
            rebano_train_time1 = time.perf_counter()
            c_initial = torch.full((1, number_of_neurons), 1/number_of_neurons).to(device)
            f_hat_rebano_train = f_hat_train[::(Nx//N_test), i][:, None]
            rebano = ReBaNO(layers_rebano, P_list, c_initial, BC_u, f_hat_rebano_train,
                    P_resid_values, P_BC_values, P_xx_term).to(device)
            losses_train[i] = rebano_train(rebano, x_test, BC_x, epochs_rebano, lr_rebano, testing=True)
            
            rebano_sol_train[:, i] = rebano(test_x=x_test).flatten()
            torch.cuda.synchronize()
            rebano_train_time2 = time.perf_counter()
            total_time += (rebano_train_time2 - rebano_train_time1)
            
        rebano_sol_train = rebano_sol_train.detach().cpu().numpy()
        np.save(datapath + f"ReBaNO_solutions_train_K_{K}_n_{n_train}_s{Nx}.npy", rebano_sol_train)
        
        print(f"avg training time on training dataset: {total_time/n_train} seconds")

        rel_err_rebano_train = np.ones(n_train)
        
        for i in range(n_train):
            rel_err_rebano_train[i] = np.linalg.norm(rebano_sol_train[:, i] - u_train[::(Nx//N_test), i])/np.linalg.norm(u_train[::(Nx//N_test), i])
        
        np.save(datapath + f"ReBaNO_rel_err_train_K_{K}_n_{n_train}_s{Nx}.npy", rel_err_rebano_train)
        print(f'mean loss on training dataset: {np.mean(losses_train)}')
        print(f'mean rel train err: {np.mean(rel_err_rebano_train)}, largest rel train err: {np.max(rel_err_rebano_train)}')

    total_time = 0
    for i in range(n_test):
        torch.cuda.synchronize()
        rebano_test_time1 = time.perf_counter()
        c_initial = torch.full((1, number_of_neurons), 1/number_of_neurons).to(device)
        f_hat_rebano_test = f_hat_test[::(Nx//N_test), i][:, None]
        rebano = ReBaNO(layers_rebano, P_list, c_initial, BC_u, f_hat_rebano_test,
                P_resid_values, P_BC_values, P_xx_term).to(device)
        losses_test[i] = rebano_train(rebano, x_test, BC_x, epochs_rebano, lr_rebano, testing=True)
        
        rebano_sol_test[:, i] = rebano(test_x=x_test).flatten() 
        torch.cuda.synchronize()
        rebano_test_time2 = time.perf_counter()
        total_time += (rebano_test_time2 - rebano_test_time1)
    rebano_sol_test = rebano_sol_test.detach().cpu().numpy()
    np.save(datapath + f"ReBaNO_solutions_test_K_{K}_n_{n_test}_s{Nx}.npy", rebano_sol_test)

    print(f"avg training time on test dataset: {total_time/n_test} seconds")

    rel_err_rebano_test  = np.ones(n_test)

    for i in range(n_test):
        rel_err_rebano_test[i] = np.linalg.norm(rebano_sol_test[:, i] - u_test[::(Nx//N_test), i])/np.linalg.norm(u_test[::(Nx//N_test), i])
        
    np.save(datapath + f"ReBaNO_rel_err_test_K_{K}_n_{n_test}_s{Nx}.npy", rel_err_rebano_test)
        
    print(f'mean loss on test dataset: {np.mean(losses_test)}')
    print(f'mean rel test err: {np.mean(rel_err_rebano_test)}, largest rel test err: {np.max(rel_err_rebano_test)} \n')