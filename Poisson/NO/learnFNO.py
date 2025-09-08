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

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print(f"current device : {device}")
################################################################
# load data and data normalization
################################################################

n_train = 1000
n_test  = 200
width = 64
batch_size = 16

Nx = 128

s = Nx



K = 10
prefix = "../data/"  
fx = np.load(prefix + "poisson1d_input_f_K_" + str(K) + "_s" + str(Nx) + ".npy")
ux = np.load(prefix + "poisson1d_output_u_K_" + str(K) + "_s" + str(Nx) + ".npy")

# transpose
ux = ux.transpose(1, 0)
fx = fx.transpose(1, 0)

x_train_ = torch.from_numpy(np.reshape(fx[:n_train, :], -1).astype(np.float32))
y_train = torch.from_numpy(np.reshape(ux[:n_train, :], -1).astype(np.float32))

# x_test = torch.from_numpy(np.reshape(fx[M//2:M, :], -1).astype(np.float32))
# y_test = torch.from_numpy(np.reshape(ux[M//2:M, :], -1).astype(np.float32))


# x_normalizer = UnitGaussianNormalizer(x_train)
# x_train = x_normalizer.encode(x_train)
# x_test = x_normalizer.encode(x_test)

# y_normalizer = UnitGaussianNormalizer(y_train)
# y_train = y_normalizer.encode(y_train)

x_grid = torch.linspace(0, 1, s+1)[:-1].reshape(-1, 1)
x_train_ = x_train_.reshape(n_train, s, 1)

x_train = torch.ones(n_train, s, 2)
for i in range(n_train):
    x_train[i] = torch.cat([x_train_[i], x_grid], dim=1)


# x_test = x_test.reshape(ntest,s,2)

# todo do we need this
y_train = y_train.reshape(n_train, s, 1)
# y_test = y_test.reshape(ntest,s,1)



################################################################
# training and evaluation
################################################################

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
# test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

layers_FNL = 2
learning_rate = 0.001
epochs = 1000
step_size = 100
gamma = 0.5

modes = 10


model = FNO1d(modes, width).to(device)
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
        out = model(x).reshape(batch_size_, s)
        # out = y_normalizer.decode(out)
        # y = y_normalizer.decode(y)

        loss = myloss(out.view(batch_size_,-1), y.view(batch_size_,-1))
        loss.backward()

        optimizer.step()
        train_l2 += loss.item()

    # torch.save(model, "../model/FNO_width_"+str(width)+"_Nd_"+str(n_train)+"_l_" + str(layers_FNL)+"_K_"+str(K)+".model")
    scheduler.step()

    train_l2/= n_train

    t2 = perf_counter()
    if ep >= 100 and ep % 100 == 0:
        print("Epoch : ", ep, " Epoch time : ", t2-t1, " Train L2 Loss : ", train_l2)

t3 = perf_counter()


print("FNO Training, # of training cases : ", n_train)
print("Total time is :", t3 - t0,"seconds")
print("Total epoch is : ", epochs)
print("Final loss is : ", train_l2)