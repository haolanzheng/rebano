import torch
import torch.nn as nn
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.optim import AdamW
from CNO1d import CNO1d
import matplotlib.pyplot as plt

from utils import LpLoss, count_params, save_ckpt
from time import perf_counter
from argparse import ArgumentParser

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.set_default_dtype(torch.float)
# In this script, we approxiamte solution of the 1d Allen-Cahn equation

def main(args):
    
    torch.manual_seed(args.seed)
    
    n_train = 1000 # number of training samples
    n_test  = 1000

    # Load the data
    # - data/AC_data_input.npy
    # - data/AC_data_output.npy

    # We will decrease the resolution to s = 256, for more convenient training
    Nx = 800
    x_data = torch.from_numpy(np.load(f"../../PIOLA/Poisson/data/Random_Poisson1d_trunc_input_f_10_{Nx}.npy")).type(torch.float32)
    y_data = torch.from_numpy(np.load(f"../../PIOLA/Poisson/data/Random_Poisson1d_trunc_output_u_10_{Nx}.npy")).type(torch.float32)

    x_data = x_data.permute(1,0)
    y_data = y_data.T.unsqueeze(1)
    
    x_data = x_data.unsqueeze(1)
    s = 128
    x_data = F.interpolate(x_data.unsqueeze(2), size = (1, s), mode = "bicubic")[:,:,0]
    y_data = F.interpolate(y_data.unsqueeze(2), size = (1, s), mode = "bicubic")[:,:,0]
    
    x_grid = torch.linspace(0, 1, steps=s+1, dtype=torch.float)[:-1]
    coords = x_grid.unsqueeze(0).repeat(x_data.shape[0], 1)
    x_data = torch.stack([x_data.squeeze(1), coords], dim=1)

    input_function_train = x_data[:n_train, :]
    output_function_train = y_data[:n_train, :]
    input_function_test = x_data[:n_test, :]
    output_function_test = y_data[:n_test, :]
    
    #---------------------
    # Define the hyperparameters and the model:
    #---------------------

    N_layers = 4
    N_res    = 2
    N_res_neck = 2
    channel_multiplier = 8

    cno = CNO1d(in_dim = 2,                                    # Number of input channels.
                out_dim = 1,                                   # Number of input channels.
                size = s,                                      # Input and Output spatial size (required )
                N_layers = N_layers,                           # Number of (D) or (U) blocks in the network
                N_res = N_res,                                 # Number of (R) blocks per level (except the neck)
                N_res_neck = N_res_neck,                       # Number of (R) blocks in the neck
                channel_multiplier = channel_multiplier,       # How the number of channels evolve?
                use_bn = False).to(device)
    print("# of params:", count_params(cno))
        
        
        
    if args.train:
        #-----------
        # TRAIN:
        #-----------
        
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
        
        save_ckpt(f'poisson-cno-{epochs}.pt', cno, optimizer, scheduler)
    
    
    
    if args.test:
        testing_set = DataLoader(TensorDataset(input_function_test, output_function_test), batch_size=1, shuffle=False)
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
                t1 = perf_counter()
                loss_f = l(output_batch, output_pred_batch)
                total_time += t1 - t0
                test_relative_l2 += loss_f.item()
            test_relative_l2 /= len(testing_set)
        print(f"Avg Inference Time : {total_time/n_test} seconds")
        print("Relative L2 Test Error:", test_relative_l2)
    
    
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
