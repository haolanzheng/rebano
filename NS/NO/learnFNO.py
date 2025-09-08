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

import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial

from time import perf_counter

torch.manual_seed(0)
np.random.seed(0)


################################################################
# load data and data normalization
################################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"current device : {device}")
# M = int(sys.argv[1]) #5000
width = 20
batch_size = 16


Nx = 100
n_train = 1000
n_test = 200
s = Nx

K = 8
prefix = "../data/"
curl_f = np.load(prefix + f"ns_input_curl_f_K_{K}_s{Nx}.npy") 
omega = np.load(prefix + f"ns_output_omega_K_{K}_s{Nx}.npy")

# transpose
curl_f = curl_f.transpose(2, 0, 1)
omega = omega.transpose(2, 0, 1)

x_train = torch.from_numpy(np.reshape(curl_f[:n_train, :, :], -1).astype(np.float32))
y_train = torch.from_numpy(np.reshape(omega[:n_train, :, :], -1).astype(np.float32))

# x_test = torch.from_numpy(np.reshape(curl_f[n_train:(n_train+n_test), :, :], -1).astype(np.float32))
# y_test = torch.from_numpy(np.reshape(omega[n_train:(n_train+n_test), :, :], -1).astype(np.float32))

x_grid = torch.linspace(0, 2*np.pi, s+1)[:-1]
x_train_ = x_train.reshape(n_train, s, s, 1)

X, Y = torch.meshgrid(x_grid, x_grid, indexing='ij')
mesh = torch.stack([X, Y], dim=2)

x_train = torch.ones(n_train, s, s, 3)
for i in range(n_train):
    x_train[i] = torch.cat([x_train_[i], mesh], dim=2)
# x_test = x_test.reshape(n_test, s, s, 1)

# todo do we need this
y_train = y_train.reshape(n_train, s, s, 1)
# y_test = y_test.reshape(n_test, s, s, 1)



################################################################
# training and evaluation
################################################################

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
# test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

learning_rate = 0.001

epochs = 1000
step_size = 100
gamma = 0.5

modes = 6
layers_FNL = 2

model = FNO2d(modes, modes, width).to(device)
print(count_params(model))

optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

myloss = LpLoss(size_average=False)

t0 = perf_counter()
for ep in range(1, epochs+1):
    model.train()
    t1 = perf_counter()
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        batch_size_ = x.shape[0]
        optimizer.zero_grad()
        out = model(x).reshape(batch_size_, s, s)

        loss = myloss(out.view(batch_size_,-1), y.view(batch_size_,-1))
        loss.backward()

        optimizer.step()
        train_l2 += loss.item()

    torch.save(model, f"../model/FNO_width_{width}_Nd_{n_train}_l_{layers_FNL}_K_{K}.model")
    scheduler.step()

    train_l2/= n_train

    t2 = perf_counter()
    if ep % 100 == 0:
        print("Epoch : ", ep, " Epoch time : ", t2-t1, " Rel. Train L2 Loss : ", train_l2)

t3 = perf_counter()

print("FNO Training, # of training cases : ", n_train)
print("Total time is : ", t3 - t0, "seconds")
print("Total epoch is : ", epochs)
print("Final loss is : ", train_l2)