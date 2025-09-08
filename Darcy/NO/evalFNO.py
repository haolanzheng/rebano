"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Darcy Flow discussed in Section 5.2 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""

import sys
sys.path.append('../../nn')
from neuralop import *
from mydata import UnitGaussianNormalizer
from Adam import Adam

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from datetime import datetime

import operator
from functools import reduce
from functools import partial

from time import perf_counter


torch.manual_seed(0)
np.random.seed(0)

n_train = 1000
n_test  = 200 
width = 32
layers_FNL = 2

Nx = 100
s = Nx

K = 8
prefix = "../data/"
a_field = np.load(prefix + f"darcy_input_a_K_{K}_s{Nx}.npy") 
u_sol   = np.load(prefix + f"darcy_output_u_K_{K}_s{Nx}.npy")

a_field_ood = np.load(prefix + f"darcy_input_a_K_{K}_s{Nx}_ood.npy") 
u_sol_ood   = np.load(prefix + f"darcy_output_u_K_{K}_s{Nx}_ood.npy")

inputs = np.copy(a_field)
outputs = np.copy(u_sol)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"current device : {device}")
################################################################
# load data and data normalization
################################################################

test_on_ood = True

# transpose
a_field = a_field.transpose(2, 0, 1)
u_sol = u_sol.transpose(2, 0, 1)

if test_on_ood:
    a_field_test = a_field_ood.transpose(2, 0, 1)
    u_sol_test = u_sol_ood.transpose(2, 0, 1)
    datapath = "../data/ood/"
else:
    a_field_test = a_field[n_train:, :, :]
    u_sol_test = u_sol[n_train:, :, :]
    datapath = "../data/"

x_train = torch.from_numpy(np.reshape(a_field[:n_train, :-1, :-1], -1).astype(np.float32))
y_train = torch.from_numpy(np.reshape(u_sol[:n_train, :-1, :-1], -1).astype(np.float32))

x_test = torch.from_numpy(np.reshape(a_field_test[:n_test, :-1, :-1], -1).astype(np.float32))
y_test = torch.from_numpy(np.reshape(u_sol_test[:n_test, :-1, :-1], -1).astype(np.float32))


# x_normalizer = UnitGaussianNormalizer(x_train)
# x_train = x_normalizer.encode(x_train)
# x_test = x_normalizer.encode(x_test)

# y_normalizer = UnitGaussianNormalizer(y_train)
# y_train = y_normalizer.encode(y_train)


x_grid = torch.linspace(0, 1, s+1)
x_grid = x_grid[0:-1]
X, Y = torch.meshgrid(x_grid, x_grid, indexing='ij')
mesh = torch.stack([X, Y], dim=2)

x_train_ = x_train.reshape(n_train,s,s,1)
x_test_ = x_test.reshape(n_test,s,s,1)
x_train = torch.ones(n_train, s, s, 3)
for i in range(n_train):
    x_train[i] = torch.cat([x_train_[i], mesh], dim=2)
x_test = torch.ones(n_test, s, s, 3)
for i in range(n_test):
    x_test[i] = torch.cat([x_test_[i], mesh], dim=2)

# todo do we need this
y_train = y_train.reshape(n_train,s,s,1)
y_test = y_test.reshape(n_test,s,s,1)


model = torch.load(f"../model/FNO_width_{width}_Nd_{n_train}_l_{layers_FNL}_K_{K}.model", map_location=device)

# Training error
y_pred_train = torch.ones(Nx, Nx, n_train, dtype=torch.float).to(device)
y_pred_test  = torch.ones(Nx, Nx, n_test, dtype=torch.float).to(device)

model.eval()
x_train = x_train.to(device)

#warm up
for i in range(100):
    # print("i / N = ", i, " / ", M//2)
    model(x_train[i:i+1, :, :, :])
    
rel_err_fno_train = np.zeros(n_train)

with torch.no_grad():
    total_time = 0
    for i in range(n_train):
        # print("i / N = ", i, " / ", M//2)
        t0 = perf_counter()
        y_pred_train[:, :, i] = model(x_train[i:i+1, :, :, :]).detach().reshape(Nx, Nx)
        t1 = perf_counter()
        total_time += (t1 - t0)
y_pred_train = y_pred_train.cpu().numpy()

np.save(datapath + f"FNO_solutions_train_K_{K}_n_{n_train}.npy", y_pred_train)
print(f'avg evaluation time on training dataset : {total_time/n_train} seconds')

# y_train = y_normalizer.decode(y_train).cpu().numpy().squeeze(-1)
y_train = y_train.cpu().numpy().squeeze(-1)
for i in range(n_train):
    rel_err_fno_train[i] =  np.linalg.norm(y_pred_train[:, :, i] - y_train[i, :, :])/np.linalg.norm(y_train[i, :, :])
mean_rel_err_train = np.mean(rel_err_fno_train)
max_rel_err_train  = np.max(rel_err_fno_train)



########### Test
x_test = x_test.to(device)
rel_err_fno_test = np.zeros(n_test)
with torch.no_grad():
    total_time = 0
    for i in range(n_test):
        # print("i / N = ", i, " / ", M-M//2)
        t0 = perf_counter()
        y_pred_test[:, :, i] = model(x_test[i:i+1, :, :, :]).detach().reshape(Nx, Nx)
        t1 = perf_counter()
        total_time += (t1 - t0)
y_pred_test = y_pred_test.cpu().numpy()

np.save(datapath + f"FNO_solutions_test_K_{K}_n_{n_test}.npy", y_pred_test)
print(f"avg evaluation time on test dataset : {total_time/n_test} seconds")

y_test = y_test.squeeze(-1)
for i in range(n_test):    
    rel_err_fno_test[i] =  np.linalg.norm(y_pred_test[:, :, i] - y_test[i, :, :].cpu().numpy())/np.linalg.norm(y_test[i, :, :].cpu().numpy())
mean_rel_err_test = np.mean(rel_err_fno_test)
max_rel_err_test  = np.max(rel_err_fno_test)

np.save(datapath + f"FNO_rel_err_train_K_{K}_n_{n_train}.npy", rel_err_fno_train)
np.save(datapath + f"FNO_rel_err_test_K_{K}_n_{n_test}.npy", rel_err_fno_test)

print(f"mean rel train err : {mean_rel_err_train}, largest rel train err : {max_rel_err_train}")
print(f"mean rel test err : {mean_rel_err_test}, largest rel test err : {max_rel_err_test}")




