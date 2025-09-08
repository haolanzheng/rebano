import sys
import numpy as np
sys.path.append('../../nn')
from neuralop import *
from mydata import UnitGaussianNormalizer
from Adam import Adam
from time import perf_counter

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"current device : {device}")


torch.manual_seed(0)
np.random.seed(0)

N_neurons = 200
layers = 4
batch_size = 16

Nx = 100  # element
n_train = 1000
n_test  = 200
K = 8
prefix = "../data/"

curl_f = np.load(prefix + f"ns_input_curl_f_K_{K}_s{Nx}.npy") 
omega = np.load(prefix + f"ns_output_omega_K_{K}_s{Nx}.npy")


acc = 0.999

xgrid = np.linspace(0, 2*np.pi, Nx+1)
xgrid = xgrid[0:-1]
# xgrid = xgrid[:-1]
dx    = xgrid[1] - xgrid[0]

inputs  = curl_f
outputs = omega

compute_input_PCA = True

if compute_input_PCA:
    train_inputs = np.reshape(inputs[:, :, :n_train], (-1, n_train))
    test_inputs  = np.reshape(inputs[:, :, n_train:(n_train+n_test)], (-1, n_test))
    Ui,Si,Vi = np.linalg.svd(train_inputs)
    en_f= 1 - np.cumsum(Si)/np.sum(Si)
    r_f = np.argwhere(en_f<(1-acc))[0,0]
    # r_f = min(r_f, 512)
    # print(Si[98:210])
    r_f = min(r_f, 512)
    Uf = Ui[:,:r_f]
    f_hat = np.matmul(Uf.T,train_inputs)
    x_train = torch.from_numpy(f_hat.T.astype(np.float32))



train_outputs = np.reshape(outputs[:, :, :n_train], (-1, n_train))
test_outputs  = np.reshape(outputs[:, :, n_train:(n_train+n_test)], (-1, n_test))
Uo,So,Vo = np.linalg.svd(train_outputs)
en_g = 1 - np.cumsum(So)/np.sum(So)
r_g = np.argwhere(en_g<(1-acc))[0,0]
Ug = Uo[:,:r_g]
g_hat = np.matmul(Ug.T,train_outputs)
y_train = torch.from_numpy(g_hat.T.astype(np.float32))


# x_normalizer = UnitGaussianNormalizer(x_train)
# x_train = x_normalizer.encode(x_train)
# y_normalizer = UnitGaussianNormalizer(y_train)
# y_train = y_normalizer.encode(y_train)

print("Input #bases : ", r_f, " output #bases : ", r_g)
 
################################################################
# training and evaluation
################################################################


train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)

learning_rate = 0.001

epochs = 1000
step_size = 100
gamma = 0.5


model = FNN(r_f, r_g, layers, N_neurons) 
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

    torch.save(model, "../model/PCANet_width_"+str(N_neurons)+"_Nd_"+str(n_train)+"_l_" + str(layers) + "_K_" + str(K) + ".model")
    scheduler.step()

    train_l2/= n_train

    t2 = perf_counter()
    if ep % 100 == 0:
        print("Epoch : ", ep, " Epoch time : ", t2-t1, " Train L2 Loss : ", train_l2)

t3 = perf_counter()

print("PCA-Net Training, # of training cases : ", n_train)
print("Total time is : ", t3 - t0, "seconds")
print("Total epoch is : ", epochs)
print("Final loss is : ", train_l2)
