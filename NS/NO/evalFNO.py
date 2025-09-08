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
width = 20
layers_FNL = 2

Nx = 100
s = Nx

K = 8
prefix = "../data/"
curl_f = np.load(prefix + f"ns_input_curl_f_K_{K}_s{Nx}.npy") 
omega = np.load(prefix + f"ns_output_omega_K_{K}_s{Nx}.npy")

curl_f_ood = np.load(prefix + f"ns_input_curl_f_K_{K}_s{Nx}_ood.npy") 
omega_ood = np.load(prefix + f"ns_output_omega_K_{K}_s{Nx}_ood.npy")

inputs = np.copy(curl_f)
outputs = np.copy(omega)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"current device : {device}")
################################################################
# load data and data normalization
################################################################
test_on_ood = True

# transpose
curl_f = curl_f.transpose(2, 0, 1)
omega = omega.transpose(2, 0, 1)

if test_on_ood:
    curl_f_test = curl_f_ood.transpose(2, 0, 1)
    omega_test = omega_ood.transpose(2, 0, 1)
    datapath = "../data/ood/"
else:
    curl_f_test = curl_f[n_train:, :, :]
    omega_test = omega[n_train:, :, :]
    datapath = "../data/"

x_train = torch.from_numpy(np.reshape(curl_f[:n_train, :, :], -1).astype(np.float32))
y_train = torch.from_numpy(np.reshape(omega[:n_train, :, :], -1).astype(np.float32))

x_test = torch.from_numpy(np.reshape(curl_f_test[:n_test, :, :], -1).astype(np.float32))
y_test = torch.from_numpy(np.reshape(omega_test[:n_test, :, :], -1).astype(np.float32))


x_train_ = x_train.reshape(n_train,s,s,1)
x_test_ = x_test.reshape(n_test,s,s,1)

# todo do we need this
y_train = y_train.reshape(n_train,s,s,1)
y_test = y_test.reshape(n_test,s,s,1)

x_grid = torch.linspace(0, 2*np.pi, s+1)
x_grid = x_grid[0:-1]
X, Y = torch.meshgrid(x_grid, x_grid, indexing='ij')
mesh = torch.stack([X, Y], dim=2)

x_train = torch.ones(n_train, s, s, 3)
for i in range(n_train):
    x_train[i] = torch.cat([x_train_[i], mesh], dim=2)

x_test = torch.ones(n_test, s, s, 3)
for i in range(n_test):
    x_test[i] = torch.cat([x_test_[i], mesh], dim=2)

model = torch.load(f"../model/FNO_width_{width}_Nd_{n_train}_l_{layers_FNL}_K_{K}.model", map_location=device)

# Training error
y_pred_train = torch.zeros(Nx, Nx, n_train, dtype=torch.float).to(device)
y_pred_test  = torch.zeros(Nx, Nx, n_test, dtype=torch.float).to(device)
# warm up
model.eval()
x_train = x_train.to(device)
for i in range(n_train):
    # print("i / N = ", i, " / ", M//2)
    model(x_train[i:i+1])
    
rel_err_fno_train = np.zeros(n_train)
with torch.no_grad():
    total_time = 0.0
    for i in range(n_train):
        t0 = perf_counter()
        y_pred_train[:, :, i] = model(x_train[i:i+1, :, :, :]).detach().reshape(Nx, Nx)
        t1 = perf_counter()
        total_time += (t1 - t0)

y_pred_train = y_pred_train.cpu().numpy()
np.save(datapath + f"FNO_solutions_train_K_{K}_n_{n_train}.npy", y_pred_train)
print(f'avg evaluation time on training dataset : {total_time/n_train} seconds')

y_train = y_train.squeeze(-1).numpy()

for i in range(n_train):
    rel_err_fno_train[i] =  np.linalg.norm(y_pred_train[:, :, i] - y_train[i, :, :])/np.linalg.norm(y_train[i, :, :])
mean_rel_err_train = np.mean(rel_err_fno_train)
max_rel_err_train  = np.max(rel_err_fno_train)



########### Test
x_test = x_test.to(device)
rel_err_fno_test = np.zeros(n_test)

with torch.no_grad():
    total_time = 0.0
    for i in range(n_test):
        t0 = perf_counter()
        y_pred_test[:, :, i] = model(x_test[i:i+1, :, :, :]).detach().reshape(Nx, Nx)
        t1 = perf_counter()
        total_time += (t1 - t0)

y_pred_test = y_pred_test.cpu().numpy()
np.save(datapath + f"FNO_solutions_test_K_{K}_n_{n_test}.npy", y_pred_test)
print(f"total evaluation time on test dataset : {t1 - t0} seconds")

y_test = y_test.squeeze(-1).numpy()
for i in range(n_test):    
    rel_err_fno_test[i] =  np.linalg.norm(y_pred_test[:, :, i] - y_test[i, :, :])/np.linalg.norm(y_test[i, :, :])
mean_rel_err_test = np.mean(rel_err_fno_test)
max_rel_err_test  = np.max(rel_err_fno_test)

np.save(datapath + f"FNO_rel_err_train_K_{K}_n_{n_train}.npy", rel_err_fno_train)
np.save(datapath + f"FNO_rel_err_test_K_{K}_n_{n_test}.npy", rel_err_fno_test)

print(f"mean rel train err : {mean_rel_err_train}, largest rel train err : {max_rel_err_train}")
print(f"mean rel test err : {mean_rel_err_test}, largest rel test err : {max_rel_err_test}")




