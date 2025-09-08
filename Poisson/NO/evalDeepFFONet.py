import sys
import numpy as np
sys.path.append('../../nn')
from neuralop import *
from mydata import *
from time import perf_counter
from datetime import datetime

torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"current device : {device}")

n_train = 1000
n_test  = 1000
N_neurons = 60
layers = 4

Nx = 128 
K = 10

prefix = "../data/"
fx = np.load(prefix + "poisson1d_input_f_K_" + str(K) + "_s" + str(Nx) + ".npy")
ux = np.load(prefix + "poisson1d_output_u_K_" + str(K) + "_s" + str(Nx) + ".npy")

fx_ood = np.load(prefix + "poisson1d_input_f_K_" + str(K) + "_s" + str(Nx) + "_ood.npy")
ux_ood = np.load(prefix + "poisson1d_output_u_K_" + str(K) + "_s" + str(Nx) + "_ood.npy")

u_train = ux[:, :n_train]

acc = 0.999

xgrid = np.linspace(0, 1, Nx+1)
xgrid = xgrid[0:-1]
dx    = xgrid[1] - xgrid[0]

inputs  = fx
outputs = ux

test_on_ood = False

if test_on_ood:
    inputs_test  = fx_ood
    outputs_test = ux_ood
    u_test       = ux_ood[:, :n_test]
    datapath = "../data/ood/"
else:
    inputs_test  = fx[:, n_train:]
    outputs_test = ux[:, n_train:]
    u_test       = ux[:, n_train:n_train+n_test]
    datapath = "../data/"

compute_input_PCA = True

if compute_input_PCA:
    train_inputs = inputs[:, :n_train]
    test_inputs  = inputs_test[:, :n_test]
    Ui,Si,Vi = np.linalg.svd(train_inputs)
    en_f= 1 - np.cumsum(Si)/np.sum(Si)
    r_f = np.argwhere(en_f<(1-acc))[0,0]

    r_f = min(r_f, 128)
    # r_f = 128

    Uf = Ui[:,:r_f]
    f_hat = np.matmul(Uf.T,train_inputs)
    f_hat_test = np.matmul(Uf.T,test_inputs)

    x_train_part = f_hat.T.astype(np.float32)
    x_test_part = f_hat_test.T.astype(np.float32)
    

del Ui, Vi, Uf, f_hat

X = xgrid
X_upper = np.reshape(X, -1)
N_upper = len(X_upper)
x_train = x_train_part
y_train = np.zeros((n_train, N_upper), dtype = np.float32)

for i in range(n_train):
    y_train[i,:] = ux[:, i]
X_upper = X_upper.reshape(-1,1)    
print("Input dim : ", r_f, " output dim : ", N_upper)


X_upper = torch.from_numpy(X_upper.astype(np.float32)).to(device)
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)


# x_normalizer = UnitGaussianNormalizer(x_train)
# x_train = x_normalizer.encode(x_train)
# y_normalizer = UnitGaussianNormalizer(y_train)


# if torch.cuda.is_available():
    # x_normalizer.gpu()
#     y_normalizer.gpu()

print("Input dim : ", r_f+1, " output dim : ", 1)
 

model = torch.load("../model/DeepFFONet_width_"+str(N_neurons)+"_Nd_"+str(n_train)+"_l_"+str(layers)+"_K_"+str(K) +".model", map_location=device)
model.to(device)

model.eval()

x_train = x_train.to(device)

for i in range(100):
    model(x_train[i:i+1])
    
u_train_pred_upper = torch.ones(Nx, n_train, dtype=torch.float).to(device)
u_test_pred_upper  = torch.ones(Nx, n_test, dtype=torch.float).to(device)

# Training error
with torch.no_grad():
    total_time = 0
    for i in range(n_train):
        torch.cuda.synchronize()
        t0 = perf_counter()
        u_train_pred_upper[:, i] = model(x_train[i:i+1]).flatten()
        torch.cuda.synchronize()
        t1 = perf_counter()
        total_time += (t1 - t0)

u_train_pred_upper = u_train_pred_upper.detach().cpu().numpy()
np.save(datapath + f"DeepONet_solutions_train_K_{K}_n_{n_train}_{Nx}.npy", u_train_pred_upper)

print(f"avg evaluation time on training dataset : {total_time/n_train} seconds")

rel_err_don_train = np.zeros(n_train)
for i in range(n_train):
    # print("i / N = ", i, " / ", M//2)
    u_train_pred = np.reshape(u_train_pred_upper[:, i],(Nx,))
    rel_err_don_train[i] =  np.linalg.norm(u_train_pred - u_train[:, i])/np.linalg.norm(u_train[:, i])
mean_rel_err_train = np.mean(rel_err_don_train)
max_rel_err_train  = np.max(rel_err_don_train)

del x_train,  u_train
########### Test
x_test = x_test_part
# x_normalizer.cpu()
x_test = torch.from_numpy(x_test)
x_test = x_test.to(device)
# Test error
rel_err_don_test = np.zeros(n_test)
with torch.no_grad():
    total_time = 0
    for i in range(n_test):
        torch.cuda.synchronize()
        t0 = perf_counter()
        u_test_pred_upper[:, i] = model(x_test[i:i+1]).flatten()
        torch.cuda.synchronize()
        t1 = perf_counter()
        total_time += (t1 - t0)

u_test_pred_upper = u_test_pred_upper.detach().cpu().numpy()
np.save(datapath + f"DeepONet_solutions_test_K_{K}_n_{n_test}_{Nx}.npy", u_test_pred_upper)

print(f"avg evaluation time on test dataset : {total_time/n_test} seconds")

for i in range(n_test):
    # print("i / N = ", i, " / ", M-M//2)
    u_test_pred = np.reshape(u_test_pred_upper[:, i], (Nx,))
    rel_err_don_test[i] =  np.linalg.norm(u_test_pred - u_test[:, i])/np.linalg.norm(u_test[:, i])
mean_rel_err_test = np.mean(rel_err_don_test)
max_rel_err_test  = np.max(rel_err_don_test)

np.save(datapath + f"DeepONet_rel_err_train_K_{K}_n_{n_train}_{Nx}.npy", rel_err_don_train)
np.save(datapath + f"DeepONet_rel_err_test_K_{K}_n_{n_test}_{Nx}.npy", rel_err_don_test)

print(f"mean rel train err : {mean_rel_err_train}, largest rel train err : {max_rel_err_train}")
print(f"mean rel test err : {mean_rel_err_test}, largest rel test err : {max_rel_err_test}")
