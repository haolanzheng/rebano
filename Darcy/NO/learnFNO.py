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
width = 32
batch_size = 16
layers_FNL = 2


Nx = 100
n_train = 1000
n_test = 200
s = Nx

x_grid = torch.linspace(0, 1, s+1)
x_grid = x_grid[0:-1]

K = 8
prefix = "../data/"
a_field = np.load(prefix + f"darcy_input_a_K_{K}_s{Nx}.npy") 
u_sol   = np.load(prefix + f"darcy_output_u_K_{K}_s{Nx}.npy")

# transpose
a_field = a_field.transpose(2, 0, 1)
u_sol = u_sol.transpose(2, 0, 1)

x_train = torch.from_numpy(np.reshape(a_field[:n_train, :-1, :-1], -1).astype(np.float32))
y_train = torch.from_numpy(np.reshape(u_sol[:n_train, :-1, :-1], -1).astype(np.float32))

# x_test = torch.from_numpy(np.reshape(a_field[n_train:(n_train+n_test), :, :], -1).astype(np.float32))
# y_test = torch.from_numpy(np.reshape(u_sol[n_train:(n_train+n_test), :, :], -1).astype(np.float32))


# x_normalizer = UnitGaussianNormalizer(x_train)
# x_train = x_normalizer.encode(x_train)
# x_test = x_normalizer.encode(x_test)

# y_normalizer = UnitGaussianNormalizer(y_train)
# y_train = y_normalizer.encode(y_train)

X, Y = torch.meshgrid(x_grid, x_grid, indexing='ij')
mesh = torch.stack([X, Y], dim=2)

x_train_ = x_train.reshape(n_train, s, s, 1)
x_train = torch.ones(n_train, s, s, 3)
for i in range(n_train):
    x_train[i] = torch.cat([x_train_[i], mesh], dim=2)
# x_test = x_test.reshape(n_test, s, s, 1)

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

modes = 10


model = FNO2d(modes, modes, width).to(device)
print(count_params(model))

optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

myloss = LpLoss(size_average=False)
# if torch.cuda.is_available():
#     y_normalizer.gpu()
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
        # out = y_normalizer.decode(out)
        # y = y_normalizer.decode(y)

        loss = myloss(out.reshape(y.shape), y)
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