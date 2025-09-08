import sys
import numpy as np

sys.path.append('../../nn')

from neuralop import *
from mydata import *
from datetime import datetime
from time import perf_counter

torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print(f"current device : {device}")

n_train = 1000
n_test  = 1000
N_neurons = 64



Nx = 128  # element
layers = 6
K = 10
prefix = "../data/"

fx = np.load(prefix + "poisson1d_input_f_K_" + str(K) + "_s" + str(Nx) + ".npy")
ux = np.load(prefix + "poisson1d_output_u_K_" + str(K) + "_s" + str(Nx) + ".npy")

fx_ood = np.load(prefix + "poisson1d_input_f_K_" + str(K) + "_s" + str(Nx) + "_ood.npy")
ux_ood = np.load(prefix + "poisson1d_output_u_K_" + str(K) + "_s" + str(Nx) + "_ood.npy")

acc = 0.999

xgrid = np.linspace(0, 1, Nx+1)
xgrid = xgrid[0:-1]
dx    = xgrid[1] - xgrid[0]

X = xgrid

test_on_ood = False

inputs  = fx
outputs = ux

if test_on_ood:
    inputs_test  = fx_ood 
    outputs_test = ux_ood
    datapath = "../data/ood/"
else:
    inputs_test  = fx[:, n_train:]
    outputs_test = ux[:, n_train:]
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

    x_train = torch.from_numpy(f_hat.T.astype(np.float32))

    
train_outputs = outputs[:, :n_train]
test_outputs  = outputs_test[:, :n_test]
Uo,So,Vo = np.linalg.svd(train_outputs)
en_g = 1 - np.cumsum(So)/np.sum(So)
r_g = np.argwhere(en_g<(1-acc))[0,0]
Ug = Uo[:,:r_g]
g_hat = np.matmul(Ug.T,train_outputs) 
y_train = torch.from_numpy(g_hat.T.astype(np.float32))



model = torch.load("../model/PCANet_width_"+str(N_neurons)+"_Nd_"+str(n_train)+"_l_"+str(layers)+ "_K_" + str(K) + ".model")
model.to(device)


# x_normalizer = UnitGaussianNormalizer(x_train)
# x_train = x_normalizer.encode(x_train)
# y_normalizer = UnitGaussianNormalizer(y_train)
# y_train = y_normalizer.encode(y_train)

# if torch.cuda.is_available():
#     x_normalizer.gpu()
#     y_normalizer.gpu()

x_train = x_train.to(device)

model.eval()

y_pred_train_pca = torch.ones(Nx, n_train, dtype=torch.float).to(device)
y_pred_test_pca  = torch.ones(Nx, n_test, dtype=torch.float).to(device)

# warm up
for i in range(100):
    model(x_train[i:i+1])
    
Ug = torch.from_numpy(Ug.astype(np.float32)).to(device)

with torch.no_grad():
    total_time = 0
    for i in range(n_train):
        torch.cuda.synchronize()
        t0 = perf_counter()
        y_pred_train = model(x_train[i:i+1]).detach().T
        y_pred_train_pca[:, i] = torch.matmul(Ug, y_pred_train).flatten()
        torch.cuda.synchronize()
        t1 = perf_counter()
        total_time += (t1 - t0)
y_pred_train_pca = y_pred_train_pca.cpu().numpy()     
    
np.save(datapath + f"PCA_solutions_train_K_{K}_n_{n_train}_{Nx}.npy", y_pred_train_pca)
print(f"avg evaluation time on training dataset : {total_time / n_train} seconds")

rel_err_pca_train = np.zeros(n_train)
for i in range(n_train):
    rel_err_pca_train[i] = np.linalg.norm(train_outputs[:, i]  - y_pred_train_pca[:, i])/np.linalg.norm(train_outputs[:, i])
mean_rel_err_train = np.mean(rel_err_pca_train)
max_rel_err_train  = np.max(rel_err_pca_train)



x_test = torch.from_numpy(f_hat_test.T.astype(np.float32))
x_test = x_test.to(device)

with torch.no_grad():
    total_time = 0
    for i in range(n_test):
        torch.cuda.synchronize()
        t0 = perf_counter()
        y_pred_test = model(x_test[i:i+1]).detach().T
        y_pred_test_pca[:, i] = torch.matmul(Ug, y_pred_test).flatten()
        torch.cuda.synchronize()
        t1 = perf_counter()
        total_time += (t1 - t0)
y_pred_test_pca = y_pred_test_pca.cpu().numpy()     
    
np.save(datapath + f"PCA_solutions_test_K_{K}_n_{n_test}_{Nx}.npy", y_pred_test_pca)
print(f"avg evaluation time on test dataset : {total_time / n_test} seconds")

rel_err_pca_test = np.zeros(n_test)
for i in range(n_test):
    rel_err_pca_test[i] = np.linalg.norm(test_outputs[:, i]  - y_pred_test_pca[:, i])/np.linalg.norm(test_outputs[:, i])
mean_rel_err_test = np.mean(rel_err_pca_test)
max_rel_err_test  = np.max(rel_err_pca_test)

np.save(datapath + f"PCA_rel_err_train_K_{K}_n_{n_train}_{Nx}.npy", rel_err_pca_train)
np.save(datapath + f"PCA_rel_err_test_K_{K}_n_{n_test}_{Nx}.npy", rel_err_pca_test)

print(f"mean rel train err : {mean_rel_err_train}, largest rel train err : {max_rel_err_train}")
print(f"mean rel test err : {mean_rel_err_test}, largest rel test err : {max_rel_err_test}")
