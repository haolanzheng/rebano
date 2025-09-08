import torch
import numpy as np

torch.set_default_dtype(torch.float)


def create_BC_data(Xi, Xf):
    ##########################################################
    BC_x  = torch.tensor([[Xi],[Xf]])
    u_min = 0.0
    u_max = 0.0
    BC_u  = torch.tensor([[u_min],[u_max]])
    ##########################################################
    return (BC_x, BC_u)

def create_residual_data(Xi, Xf, Nc, N_test):
    ##########################################################
    x_resid = torch.linspace(Xi, Xf, Nc+1)
    x_resid = x_resid[0:-1][:, None]
    ##########################################################
    x_test = torch.linspace(Xi, Xf, N_test+1)
    x_test = x_test[0:-1][:, None]
    ##########################################################
    return (x_resid, x_test)