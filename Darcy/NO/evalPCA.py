import sys
import numpy as np
sys.path.append('../../nn')
from neuralop import *
from mydata import *
from datetime import datetime

from time import perf_counter

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"current device : {device}")

torch.manual_seed(0)
np.random.seed(0)

N_neurons = 200
layers = 5


Nx = 100  # element
n_train = 1000
n_test  = 200
K = 8
prefix = "../data/"

a_field = np.load(prefix + f"darcy_input_a_K_{K}_s{Nx}.npy") 
u_sol   = np.load(prefix + f"darcy_output_u_K_{K}_s{Nx}.npy")

a_field_ood = np.load(prefix + f"darcy_input_a_K_{K}_s{Nx}_ood.npy") 
u_sol_ood   = np.load(prefix + f"darcy_output_u_K_{K}_s{Nx}_ood.npy")

acc = 0.999

xgrid = np.linspace(0, 1, Nx+1)
dx    = xgrid[1] - xgrid[0]

inputs  = a_field[:-1, :-1, :]
outputs = u_sol[:-1, :-1, :]

test_on_ood = True

if test_on_ood:
    inputs_test  = a_field_ood[:-1, :-1, :]
    outputs_test = u_sol_ood[:-1, :-1, :]
    datapath = "../data/ood/"
else:
    inputs_test  = a_field[:-1, :-1, n_train:]
    outputs_test = u_sol[:-1, :-1, n_train:]
    datapath = "../data/"


compute_input_PCA = True

if compute_input_PCA:
    train_inputs = np.reshape(inputs[:, :, :n_train], (-1, n_train))
    test_inputs  = np.reshape(inputs_test[:, :, :n_test], (-1, n_test))
    Ui,Si,Vi = np.linalg.svd(train_inputs)
    en_f= 1 - np.cumsum(Si)/np.sum(Si)
    r_f = np.argwhere(en_f<(1-acc))[0,0]
    
    # r_f = min(r_f, 512)
    r_f = min(r_f, 512)
    
    Uf = Ui[:,:r_f]
    f_hat = np.matmul(Uf.T,train_inputs)
    f_hat_test = np.matmul(Uf.T,test_inputs)

    x_train = torch.from_numpy(f_hat.T.astype(np.float32))

    
train_outputs = np.reshape(outputs[:, :, :n_train], (-1, n_train))
test_outputs  = np.reshape(outputs_test[:, :, :n_test], (-1, n_test))
Uo,So,Vo = np.linalg.svd(train_outputs)
en_g = 1 - np.cumsum(So)/np.sum(So)
r_g = np.argwhere(en_g<(1-acc))[0,0]
Ug = Uo[:,:r_g]
g_hat = np.matmul(Ug.T,train_outputs) 
y_train = torch.from_numpy(g_hat.T.astype(np.float32))

train_outputs = train_outputs.reshape(Nx, Nx, n_train)
test_outputs  = test_outputs.reshape(Nx, Nx, n_test)



model = torch.load(f"../model/PCANet_width_{N_neurons}_Nd_{n_train}_l_{layers}_K_{K}.model")
model.to(device)


# x_normalizer = UnitGaussianNormalizer(x_train)
# x_train = x_normalizer.encode(x_train)
# y_normalizer = UnitGaussianNormalizer(y_train)
# y_train = y_normalizer.encode(y_train)

# if torch.cuda.is_available():
#     x_normalizer.gpu()
#     y_normalizer.gpu()

x_train = x_train.to(device)
Ug = torch.from_numpy(Ug).type(torch.float).to(device)

y_pred_train = torch.zeros(Nx, Nx, n_train, dtype=torch.float).to(device)
y_pred_test  = torch.zeros(Nx, Nx, n_test, dtype=torch.float).to(device)

model.eval()

# warm up
for i in range(100):
    model(x_train[i:i+1])

with torch.no_grad():
    total_time = 0
    for i in range(n_train):
        t0 = perf_counter()
        y_pred = model(x_train[i:i+1]).detach().T
        y_pred_train[:, :, i] = torch.matmul(Ug, y_pred).reshape(Nx, Nx)
        t1 = perf_counter()
        total_time += (t1 - t0)
    
y_pred_train = y_pred_train.cpu().numpy()


np.save(datapath + f"PCA_solutions_train_K_{K}_n_{n_train}.npy", y_pred_train)
print(f"avg evaluation time on training dataset : {total_time/n_train} seconds")

rel_err_pca_train = np.zeros(n_train)
for i in range(n_train):
    rel_err_pca_train[i] = np.linalg.norm(train_outputs[:, :, i]  - y_pred_train[:, :, i])/np.linalg.norm(train_outputs[:, :, i])
mean_rel_err_train = np.mean(rel_err_pca_train)
max_rel_err_train  = np.max(rel_err_pca_train)

# rel_err_nn_train = np.sum((y_pred_train-g_hat)**2,0)/np.sum(g_hat**2,0)
# mre_nn_train = np.mean(rel_err_nn_train)

# print(f_hat_test.shape)
# f_hat_test = np.matmul(Uf.T,test_inputs)
x_test = torch.from_numpy(f_hat_test.T.astype(np.float32)).to(device)

with torch.no_grad():
    total_time = 0
    for i in range(n_test):
        t0 = perf_counter()
        y_pred = model(x_test[i:i+1]).detach().T
        y_pred_test[:, :, i] = torch.matmul(Ug, y_pred).reshape(Nx, Nx)
        t1 = perf_counter()
        total_time += (t1 - t0)

y_pred_test = y_pred_test.cpu().numpy()

np.save(datapath + f"PCA_solutions_test_K_{K}_n_{n_test}.npy", y_pred_test)
print(f"avg evaluation time on test dataset : {total_time/n_test} seconds")

rel_err_pca_test = np.zeros(n_test)
for i in range(n_test):
    rel_err_pca_test[i] = np.linalg.norm(test_outputs[:, :, i]  - y_pred_test[:, :, i])/np.linalg.norm(test_outputs[:, :, i])
mean_rel_err_test = np.mean(rel_err_pca_test)
max_rel_err_test  = np.max(rel_err_pca_test)

np.save(datapath + f"PCA_rel_err_train_K_{K}_n_{n_train}.npy", rel_err_pca_train)
np.save(datapath + f"PCA_rel_err_test_K_{K}_n_{n_test}.npy", rel_err_pca_test)

print(f"mean rel train err : {mean_rel_err_train}, largest rel train err : {max_rel_err_train}")
print(f"mean rel test err : {mean_rel_err_test}, largest rel test err : {max_rel_err_test}")
