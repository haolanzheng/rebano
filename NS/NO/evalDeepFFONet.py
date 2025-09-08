import sys
import numpy as np
sys.path.append('../../nn')
from neuralop import *
from mydata import *
from datetime import datetime
from time import perf_counter


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"current device : {device}")

N_neurons = 100
layers = 4

Nx = 100  # element
n_train = 1000
n_test  = 200
K = 8
prefix = "../data/"

curl_f = np.load(prefix + f"ns_input_curl_f_K_{K}_s{Nx}.npy") 
omega = np.load(prefix + f"ns_output_omega_K_{K}_s{Nx}.npy")

curl_f_ood = np.load(prefix + f"ns_input_curl_f_K_{K}_s{Nx}_ood.npy") 
omega_ood = np.load(prefix + f"ns_output_omega_K_{K}_s{Nx}_ood.npy")

omega_train = omega[:, :, :n_train]

acc = 0.999

xgrid = np.linspace(0,2*np.pi, Nx+1)
xgrid = xgrid[0:-1]
dx    = xgrid[1] - xgrid[0]

inputs  = curl_f
outputs = omega

test_on_ood = True

if test_on_ood:
    inputs_test  = curl_f_ood
    outputs_test = omega_ood
    omega_test       = omega_ood[:, :, :n_test]
    datapath = "../data/ood/"
else:
    inputs_test  = curl_f[:, :, n_train:]
    outputs_test = omega[:, :, n_train:]
    omega_test       = omega[:, :, n_train:n_train+n_test]
    datapath = "../data/"
    
compute_input_PCA = True

if compute_input_PCA:
    train_inputs = np.reshape(inputs[:,:,:n_train], (-1, n_train))
    test_inputs  = np.reshape(inputs_test[:,:,:n_test], (-1, n_test))
    Ui,Si,Vi = np.linalg.svd(train_inputs)
    en_f= 1 - np.cumsum(Si)/np.sum(Si)
    r_f = np.argwhere(en_f<(1-acc))[0,0]

    r_f = min(r_f, 512)

    Uf = Ui[:,:r_f]
    f_hat = np.matmul(Uf.T,train_inputs)
    f_hat_test = np.matmul(Uf.T,test_inputs)

    x_train_part = f_hat.T.astype(np.float32)
    x_test_part = f_hat_test.T.astype(np.float32)

del Ui, Vi, Uf, f_hat

Y, X = np.meshgrid(xgrid, xgrid)

X_upper = np.reshape(X, -1)
Y_upper = np.reshape(Y, -1)
N_upper = len(X_upper)
x_train = x_train_part
y_train = np.zeros((n_train, N_upper), dtype = np.float32)

for i in range(n_train):
    y_train[i,:] = np.reshape(omega_train[:, :, i], -1)
XY_upper = np.vstack((X_upper, Y_upper)).T    

XY_upper = torch.from_numpy(XY_upper.astype(np.float32)).to(device)
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)
    
print("Input dim : ", r_f+2, " output dim : ", 2)
 

model = torch.load(f"../model/DeepFFONet_width_{N_neurons}_Nd_{n_train}_l_{layers}_K_{K}.model", map_location=device)
model.to(device)

model.eval()
x_train = x_train.to(device)
# warm up
for i in range(100):
    model( x_train[i:i+1] )
    
y_pred_train = torch.zeros(Nx, Nx, n_train, dtype=torch.float).to(device)
y_pred_test  = torch.zeros(Nx, Nx, n_test, dtype=torch.float).to(device)

# Training error
with torch.no_grad():
    total_time = 0.0
    for i in range(n_train):
        t0 = perf_counter()
        y_pred_train[:, :, i] = model( x_train[i:i+1] ).detach().reshape(Nx, Nx)
        t1 = perf_counter()
        total_time += (t1 - t0)
y_pred_train = y_pred_train.cpu().numpy()

np.save(datapath + f"DeepONet_solutions_train_K_{K}_n_{n_train}.npy", y_pred_train)

print(f"avg evaluation time on training dataset : {total_time/n_train} seconds")

rel_err_don_train = np.zeros(n_train)
for i in range(n_train):
    # print("i / N = ", i, " / ", M//2)
    rel_err_don_train[i] =  np.linalg.norm(y_pred_train[:, :, i] - omega_train[:, :, i])/np.linalg.norm(omega_train[:, :, i])
mean_rel_err_train = np.mean(rel_err_don_train)
max_rel_err_train  = np.max(rel_err_don_train)



del x_train,  omega_train
########### Test
x_test = x_test_part
# x_normalizer.cpu()
x_test = torch.from_numpy(x_test).to(device)
# Test error
rel_err_don_test = np.zeros(n_test)
with torch.no_grad():
    total_time = 0.0
    for i in range(n_test):
        t0 = perf_counter()
        y_pred_test[:, :, i] = model(x_test[i:i+1]).detach().reshape(Nx, Nx)
        t1 = perf_counter()
        total_time += (t1 - t0)
y_pred_test = y_pred_test.cpu().numpy()

np.save(datapath + f"DeepONet_solutions_test_K_{K}_n_{n_test}.npy", y_pred_test)
print(f"avg evaluation time on test dataset : {total_time/n_test} seconds")

for i in range(n_test):
    # print("i / N = ", i, " / ", M-M//2)
    rel_err_don_test[i] =  np.linalg.norm(y_pred_test[:, :, i] - omega_test[:, :, i])/np.linalg.norm(omega_test[:, :, i])
mean_rel_err_test = np.mean(rel_err_don_test)
max_rel_err_test  = np.max(rel_err_don_test)

np.save(datapath + f"DeepONet_rel_err_train_K_{K}_n_{n_train}.npy", rel_err_don_train)
np.save(datapath + f"DeepONet_rel_err_test_K_{K}_n_{n_test}.npy", rel_err_don_test)

print(f"mean rel train err : {mean_rel_err_train}, largest rel train err : {max_rel_err_train}")
print(f"mean rel test err : {mean_rel_err_test}, largest rel test err : {max_rel_err_test}")

