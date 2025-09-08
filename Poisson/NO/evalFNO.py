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
from time import perf_counter

import operator
from functools import reduce
from functools import partial



torch.manual_seed(0)
np.random.seed(0)

n_train = 1000
n_test  = 1000
width = 64
layers_FNL = 2

Nx = 128

s = Nx

K = 10
prefix = "../data/"
fx = np.load(prefix + "poisson1d_input_f_K_" + str(K) + "_s" + str(Nx) + ".npy")
ux = np.load(prefix + "poisson1d_output_u_K_" + str(K) + "_s" + str(Nx) + ".npy")

fx_ood = np.load(prefix + "poisson1d_input_f_K_" + str(K) + "_s" + str(Nx) + "_ood.npy")
ux_ood = np.load(prefix + "poisson1d_output_u_K_" + str(K) + "_s" + str(Nx) + "_ood.npy")

inputs = np.copy(fx)
outputs = np.copy(ux)

xgrid = np.linspace(0, 1.0, Nx+1)
xgrid = xgrid[0:-1]
dx    = xgrid[1] - xgrid[0]
X = xgrid

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"current device : {device}")

batch_size = 16
learning_rate = 0.001

epochs = 1000
step_size = 100
gamma = 0.5

modes = 10



################################################################
# load data and data normalization
################################################################


# transpose
fx = fx.transpose(1, 0)
ux = ux.transpose(1, 0)
 
test_on_ood = False

if test_on_ood:
    fx_test = fx_ood.transpose(1, 0)
    ux_test = ux_ood.transpose(1, 0)
    datapath = "../data/ood/"
else:
    fx_test = fx[n_train:, :]
    ux_test = ux[n_train:, :]
    datapath = "../data/"

x_train_ = torch.from_numpy(np.reshape(fx[:n_train, :], -1).astype(np.float32))
y_train = torch.from_numpy(np.reshape(ux[:n_train, :], -1).astype(np.float32))

x_test_ = torch.from_numpy(np.reshape(fx_test[:n_test, :], -1).astype(np.float32))
y_test = torch.from_numpy(np.reshape(ux_test[:n_test, :], -1).astype(np.float32))


# x_normalizer = UnitGaussianNormalizer(x_train)
# x_train = x_normalizer.encode(x_train)
# x_test = x_normalizer.encode(x_test)

# y_normalizer = UnitGaussianNormalizer(y_train)
# y_train = y_normalizer.encode(y_train)

x_grid = torch.linspace(0, 1, s+1)[:-1].reshape(-1, 1)

x_train_ = x_train_.reshape(n_train,s,1)
x_test_  = x_test_.reshape(n_test,s,1)

x_train = torch.ones(n_train, s, 2)
x_test  = torch.ones(n_test, s, 2)
for i in range(n_train):
    x_train[i] = torch.cat([x_train_[i], x_grid], dim=1)
for i in range(n_test):
    x_test[i] = torch.cat([x_test_[i], x_grid], dim=1)



# todo do we need this
y_train = y_train.reshape(n_train,s,1)
y_test = y_test.reshape(n_test,s,1)


model = torch.load("../model/FNO_width_"+str(width)+"_Nd_"+str(n_train) + "_l_" + str(layers_FNL)+"_K_" + str(K) + ".model", map_location=device)

u_pred_train = torch.ones(Nx, n_train, dtype=torch.float).to(device)
u_pred_test  = torch.ones(Nx, n_test, dtype=torch.float).to(device)

model.eval()
x_train = x_train.to(device)
# warm up
for i in range(100):
    # print("i / N = ", i, " / ", M//2)
    model(x_train[i:i+1, :, :])

# Training error 

with torch.no_grad():
    total_time = 0
    for i in range(n_train):
        # print("i / N = ", i, " / ", M//2)
        torch.cuda.synchronize()
        t0 = perf_counter()
        u_pred_train[:, i] = model(x_train[i:i+1]).flatten()
        torch.cuda.synchronize()
        t1 = perf_counter()
        total_time += (t1 - t0)

u_pred_train = u_pred_train.detach().cpu().numpy()

print(f"avg evaluation time on training dataset : {total_time / n_train} seconds")
np.save(datapath + f"FNO_solutions_train_K_{K}_n_{n_train}_{Nx}.npy", u_pred_train)

rel_err_fno_train = np.zeros(n_train) 
for i in range(n_test):
    rel_err_fno_train[i] =  np.linalg.norm(u_pred_train[:, i] - y_train[i, :].cpu().flatten().numpy())/np.linalg.norm(y_train[i, :].cpu().flatten().numpy())
mean_rel_err_train = np.mean(rel_err_fno_train)
max_rel_err_train  = np.max(rel_err_fno_train)

########### Test
x_test = x_test.to(device)
rel_err_fno_test = np.zeros(n_test)

with torch.no_grad():
    total_time = 0
    for i in range(n_test):
        torch.cuda.synchronize()
        t0 = perf_counter()
        # print("i / N = ", i, " / ", M-M//2)
        u_pred_test[:, i] = model(x_test[i:i+1]).flatten()
        torch.cuda.synchronize()
        t1 = perf_counter()
        total_time += (t1 - t0)
    
u_pred_test = u_pred_test.detach().cpu().numpy()

np.save(datapath + f"FNO_solutions_test_K_{K}_n_{n_test}_{Nx}.npy", u_pred_test)
print(f"avg evaluation time on test dataset : {total_time / n_test} seconds")

for i in range(n_test):
    rel_err_fno_test[i] =  np.linalg.norm(u_pred_test[:, i] - y_test[i, :].cpu().flatten().numpy())/np.linalg.norm(y_test[i, :].cpu().flatten().numpy())
mean_rel_err_test = np.mean(rel_err_fno_test)
max_rel_err_test  = np.max(rel_err_fno_test)

np.save(datapath + f"FNO_rel_err_train_K_{K}_n_{n_train}_{Nx}.npy", rel_err_fno_train)
np.save(datapath + f"FNO_rel_err_test_K_{K}_n_{n_test}_{Nx}.npy", rel_err_fno_test)

print(f"mean rel train err : {mean_rel_err_train}, largest rel train err : {max_rel_err_train}")
print(f"mean rel test err : {mean_rel_err_test}, largest rel test err : {max_rel_err_test}")



