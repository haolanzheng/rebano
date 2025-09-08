import torch
import torch.nn as nn
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.optim import AdamW
from CNO2d_vanilla_torch_version.CNO2d import CNO2d
import matplotlib.pyplot as plt

from utils import LpLoss, count_params, save_ckpt
from time import perf_counter
from argparse import ArgumentParser

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.set_default_dtype(torch.float)


# In this script, we approxiamte solution of the 1d Allen-Cahn equation

def main(args):
    torch.manual_seed(args.seed)
    
    #--------------------------------------
    # REPLACE THIS PART BY YOUR DATALOADER
    #--------------------------------------
    
    n_train = 1000 # number of training samples
    n_test  = 200

    
    Nx = 100
    x_data = torch.from_numpy(np.load(f"../Darcy/data/darcy_input_a_K_8_s{Nx}.npy"))
    y_data = torch.from_numpy(np.load(f"../Darcy/data/darcy_output_u_K_8_s{Nx}.npy"))
    
    x_data = x_data.permute(2, 0, 1)
    y_data = y_data.permute(2, 0, 1).unsqueeze(1)
    
    # Add channel dimension and ensure float32 dtype
    x_data = x_data.unsqueeze(1).float()  # (batch_size, 1, H, W)
    y_data = y_data.float()               # already has channel dimension

    # Resize spatial dimensions to (s, s) using bicubic interpolation
    s = 100
    x_data = F.interpolate(x_data, size=(s, s), mode='bicubic')
    y_data = F.interpolate(y_data, size=(s, s), mode='bicubic')

    x_grid = torch.linspace(0, 1, steps=s+1, dtype=torch.float)[:-1]
    X, Y   = torch.meshgrid(x_grid, x_grid, indexing='xy')
    X, Y   = X.unsqueeze(0).repeat(x_data.shape[0], 1, 1), Y.unsqueeze(0).repeat(x_data.shape[0], 1, 1)
    coords = torch.stack([X, Y], dim=1)
    x_data = torch.cat([x_data, coords], dim=1)

    input_function_train = x_data[:n_train, :]
    output_function_train = y_data[:n_train, :]
    input_function_test = x_data[n_train:n_train+n_test, :]
    output_function_test = y_data[n_train:n_train+n_test, :]

    
    #---------------------
    # Define the hyperparameters and the model:
    #---------------------

    N_layers = 4
    N_res    = 4
    N_res_neck = 2
    channel_multiplier = 8
    
    cno = CNO2d(in_dim = 3,                                    # Number of input channels.
                out_dim = 1,                                   # Number of input channels.
                size = s,                                      # Input and Output spatial size (required )
                N_layers = N_layers,                           # Number of (D) or (U) blocks in the network
                N_res = N_res,                                 # Number of (R) blocks per level (except the neck)
                N_res_neck = N_res_neck,                       # Number of (R) blocks in the neck
                channel_multiplier = channel_multiplier,       # How the number of channels evolve?
                use_bn = False).to(device)
    print("# of params:", count_params(cno))
        
    
    data_path = './data/'
        
    if args.train:
        #-----------
        # TRAIN:
        #-----------
        learning_rate = 0.001
        epochs = 1000
        step_size = 100
        gamma = 0.5
        batch_size = 16

        # print(input_function_train.shape, output_function_train.shape)
        training_set = DataLoader(TensorDataset(input_function_train, output_function_train), batch_size=batch_size, shuffle=True)
    
        learning_rate = 0.001
        epochs = 1000
        step_size = 100
        gamma = 0.5
        optimizer = AdamW(cno.parameters(), lr=learning_rate, weight_decay=1e-8)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        
        l = LpLoss()
        cno.train()
        freq_print = 100
        train_time = 0.0
        for epoch in range(1, epochs+1):
            t0 = perf_counter()
            train_mse = 0.0
            for step, (input_batch, output_batch) in enumerate(training_set):
                input_batch, output_batch = input_batch.to(device), output_batch.to(device)
                optimizer.zero_grad()
                output_pred_batch = cno(input_batch)
                loss_f = l(output_pred_batch, output_batch)
                loss_f.backward()
                optimizer.step()
                train_mse += loss_f.item()
            train_mse /= len(training_set)

            scheduler.step()
            
            t1 = perf_counter()
            if epoch % freq_print == 0: 
                print("######### Epoch:", epoch, " ######### Train Loss:", train_mse, " ######## ep time:", t1 - t0)
                # print("######### Relative L1 Test Norm:", test_relative_l2)
            train_time += (t1 - t0)
        print(f'Training Done! Total training time : {train_time} seconds')
        
        save_ckpt(f'ckpts/darcy-cno-{epochs}.pt', cno, optimizer, scheduler)
        
        testing_set = DataLoader(TensorDataset(input_function_train, output_function_train), batch_size=1, shuffle=False)
        
        cno_pred_sol_train = torch.zeros(s, s, input_function_train.shape[0], dtype=torch.float).to(device)
        cno_rel_err_train  = np.zeros(input_function_train.shape[0])

        with torch.no_grad():
            cno.eval()
            l = LpLoss()
            total_time = 0.0
            test_relative_l2 = 0.0
            for step, (input_batch, output_batch) in enumerate(testing_set):
                t0 = perf_counter()
                input_batch, output_batch = input_batch.to(device), output_batch.to(device)
                output_pred_batch = cno(input_batch)
                # print(output_pred_batch.shape)
                t1 = perf_counter()
                loss_f = l(output_batch, output_pred_batch)
                total_time += t1 - t0
                cno_pred_sol_train[:, :, step] = output_pred_batch.reshape(s, s)
                cno_rel_err_train[step] = loss_f.item()
                test_relative_l2 += loss_f.item()
            test_relative_l2 /= len(testing_set)
        
        cno_pred_sol_train = cno_pred_sol_train.cpu().numpy()
        np.save(data_path + f'CNO_Darcy_solutions_train_K_8_n_{n_train}.npy', cno_pred_sol_train)
        np.save(data_path + f'CNO_Darcy_rel_err_train_K_8_n_{n_train}.npy', cno_rel_err_train)
        print(f"Avg Inference Time : {total_time/n_train} seconds")
        print("Mean Relative L2 Train Error:", np.mean(cno_rel_err_train))
        print("Max Relative L2 Train Error:", np.max(cno_rel_err_train))
    
    
    
    if args.test:
        testing_set = DataLoader(TensorDataset(input_function_test, output_function_test), batch_size=1, shuffle=False)
        
        cno_pred_sol_test = torch.zeros(s, s, input_function_test.shape[0], dtype=torch.float).to(device)
        cno_rel_err_test  = np.zeros(input_function_test.shape[0])
        
        if args.ckpt:
            ckpt_path = args.ckpt
            ckpt = torch.load(ckpt_path)
            cno.load_state_dict(ckpt['model'])
            print('Weights loaded from %s' % ckpt_path)

        with torch.no_grad():
            cno.eval()
            l = LpLoss()
            total_time = 0.0
            test_relative_l2 = 0.0
            for step, (input_batch, output_batch) in enumerate(testing_set):
                t0 = perf_counter()
                input_batch, output_batch = input_batch.to(device), output_batch.to(device)
                output_pred_batch = cno(input_batch)
                # print(output_pred_batch.shape)
                t1 = perf_counter()
                loss_f = l(output_batch, output_pred_batch)
                total_time += t1 - t0
                cno_pred_sol_test[:, :, step] = output_pred_batch.reshape(s, s)
                cno_rel_err_test[step] = loss_f.item()
                test_relative_l2 += loss_f.item()
            test_relative_l2 /= len(testing_set)
        
        cno_pred_sol_test = cno_pred_sol_test.cpu().numpy()
        np.save(data_path + f'CNO_Darcy_solutions_test_K_8_n_{n_test}.npy', cno_pred_sol_test)
        np.save(data_path + f'CNO_Darcy_rel_err_test_K_8_n_{n_test}.npy', cno_rel_err_test)
        print(f"Avg Inference Time : {total_time/n_test} seconds")
        print("Mean Relative L2 Test Error:", np.mean(cno_rel_err_test))
        print("Max Relative L2 Test Error:", np.max(cno_rel_err_test))
    
    
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    # parse options
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--train', action='store_true', help='Train')
    parser.add_argument('--test', action='store_true', help='Test')
    args = parser.parse_args()
    main(args)
