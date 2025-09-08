import sys
import numpy as np
sys.path.append('../../nn')
from neuralop import *
from mydata import *
from Adam import Adam


import operator
from functools import reduce
from functools import partial

from time import perf_counter



torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"current device : {device}")

N_neurons = 100
layers = 4
batch_size = 16

Nx = 100  
n_train = 1000
n_test  = 200
K = 8
prefix = "../data/"

curl_f = np.load(prefix + f"ns_input_curl_f_K_{K}_s{Nx}.npy") 
omega = np.load(prefix + f"ns_output_omega_K_{K}_s{Nx}.npy")


acc = 0.999

xgrid = np.linspace(0,2*np.pi, Nx+1)
xgrid = xgrid[0:-1]
dx    = xgrid[1] - xgrid[0]

inputs  = curl_f
outputs = omega

compute_input_PCA = True

if compute_input_PCA:
    train_inputs = np.reshape(inputs[:,:,:n_train], (-1, n_train))
    # test_inputs  = np.reshape(inputs[:,:,M//2:M], (-1, M-M//2))
    Ui,Si,Vi = np.linalg.svd(train_inputs)
    en_f= 1 - np.cumsum(Si)/np.sum(Si)
    r_f = np.argwhere(en_f<(1-acc))[0,0]
    r_f = min(r_f, 512)
    print("Energy is ", en_f[r_f - 1])
    Uf = Ui[:,:r_f]
    f_hat = np.matmul(Uf.T,train_inputs)
    x_train_part = f_hat.T.astype(np.float32)

del train_inputs
del inputs
del Ui, Vi, Uf, f_hat


Y, X = np.meshgrid(xgrid, xgrid)

X_upper = np.reshape(X, -1)
Y_upper = np.reshape(Y, -1)
N_upper = len(X_upper)
x_train = x_train_part
y_train = np.zeros((n_train, N_upper), dtype = np.float32)

for i in range(n_train):
    y_train[i, :] = np.reshape(omega[:, :, i], -1)
    
XY_upper = np.vstack((X_upper, Y_upper)).T    


print("Input dim : ", r_f+2, " output dim : ", 2)
 


XY_upper = torch.from_numpy(XY_upper.astype(np.float32)).to(device)
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

################################################################
# training and evaluation
################################################################

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)


learning_rate = 0.001

epochs = 1000
step_size = 100
gamma = 0.5



model = DeepFFONet(r_f, 2, XY_upper, layers,  layers, N_neurons) 
print(count_params(model))
model.to(device)


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
        out = model(x)

        loss = myloss(out , y)
        loss.backward()

        optimizer.step()
        train_l2 += loss.item()

    torch.save(model, f"../model/DeepFFONet_width_{N_neurons}_Nd_{n_train}_l_{layers}_K_{K}.model")
    scheduler.step()

    train_l2/= n_train

    t2 = perf_counter()
    if ep % 100 == 0:
        print("Epoch : ", ep, " Epoch time : ", t2-t1, " Train L2 Loss : ", train_l2)

t3 = perf_counter()

print("DeepONet Training, # of training cases : ", n_train)
print("Total time is : ", t3 - t0, "seconds")
print("Total epoch is : ", epochs)
print("Final loss is : ", train_l2)


