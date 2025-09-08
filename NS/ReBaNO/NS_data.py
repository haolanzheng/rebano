from torch import tensor, linspace, meshgrid, hstack, zeros, vstack
import torch

def create_ICBC_data(Xi, Xf, Yi, Yf, Ti, Tf, Nt, BC_pts, IC_pts):
    ##########################################################
    x_BC = linspace(Xi, Xf, BC_pts)
    y_BC = linspace(Yi, Yf, BC_pts)
    t_BC = linspace(Ti, Tf, Nt + 1)
    t_BC = t_BC[:, None]
    T_BC = t_BC.repeat(1, BC_pts)
    T_BC = T_BC.flatten()[:, None]
    X_BC, Y_BC = meshgrid(x_BC, y_BC, indexing='ij')
    
    x_IC = linspace(Xi, Xf*(1.0-1.0/IC_pts), IC_pts)
    y_IC = linspace(Yi, Yf*(1.0-1.0/IC_pts), IC_pts)
    
    # t_IC = linspace(Ti, Ti, IC_pts)
    # X_IC, T_IC = meshgrid(x_IC, t_IC, indexing='ij')
    ##########################################################
    XX_IC, YY_IC = meshgrid(x_IC, y_IC, indexing='xy')
    IC_x = XX_IC.flatten()[:,None]
    IC_y = YY_IC.flatten()[:,None]
    IC_t = torch.full((IC_x.shape[0], 1), Ti)     
    IC   = hstack((IC_x, IC_y))
    IC   = hstack((IC, IC_t))
    ##########################################################
    BC_left_x = X_BC[0,:][:,None] 
    BC_left_y = Y_BC[0,:][:,None] 
    
    BC_left_xy  = hstack((BC_left_x, BC_left_y)) 
    BC_left_xy  = BC_left_xy.repeat(t_BC.shape[0], 1)
    BC_left_xyt = hstack((BC_left_xy, T_BC))
    ##########################################################
    BC_right_x = X_BC[-1,:][:,None] 
    BC_right_y = Y_BC[-1,:][:,None] 
    
    BC_right_xy  = hstack((BC_right_x, BC_right_y)) 
    BC_right_xy  = BC_right_xy.repeat(t_BC.shape[0], 1)
    BC_right_xyt = hstack((BC_right_xy, T_BC))
    ##########################################################
    BC_bottom_x = X_BC[:,0][:,None] 
    BC_bottom_y = Y_BC[:,0][:,None] 
    
    BC_bottom_xy  = hstack((BC_bottom_x, BC_bottom_y)) 
    BC_bottom_xy  = BC_bottom_xy.repeat(t_BC.shape[0], 1)
    BC_bottom_xyt = hstack((BC_bottom_xy, T_BC))
    ##########################################################
    BC_top_x = X_BC[:,-1][:,None] 
    BC_top_y = Y_BC[:,-1][:,None] 
    
    BC_top_xy  = hstack((BC_top_x, BC_top_y)) 
    BC_top_xy  = BC_top_xy.repeat(t_BC.shape[0], 1)
    BC_top_xyt = hstack((BC_top_xy, T_BC))
    ##########################################################
    return (IC, BC_bottom_xyt, BC_top_xyt, BC_left_xyt, BC_right_xyt)  

def create_residual_data(Xi, Xf, Yi, Yf, Ti, Tf, Nc, Nt, N_test):
    ##########################################################
    x_resid = linspace(Xi, Xf*(1.0-1.0/Nc), Nc)
    y_resid = linspace(Yi, Yf*(1.0-1.0/Nc), Nc)
    t_resid = linspace(Ti, Tf, Nt + 1)
    
    XX_resid, YY_resid = meshgrid((x_resid, y_resid), indexing='xy')
    
    X_resid = XX_resid.flatten()[:,None]
    Y_resid = YY_resid.flatten()[:,None]
    
    xy_resid = hstack((X_resid, Y_resid))
    T_resid  = t_resid[:, None].repeat(1, xy_resid.shape[0])
    xy_resid = xy_resid.repeat(T_resid.shape[0], 1)
    T_resid  = T_resid.flatten()[:, None]
    
    
    xyt_resid = hstack((xy_resid, T_resid))
    # f_hat_train = zeros((xt_resid.shape[0], 1))
    ##########################################################
    x_test = linspace(Xi, Xf*(1.0-1.0/N_test), N_test)
    y_test = linspace(Xi, Xf*(1.0-1.0/N_test), N_test)
    t_test = linspace(Ti, Tf, Nt + 1)
    
    XX_test, YY_test = meshgrid((x_test, y_test), indexing='xy')
    
    X_test = XX_test.flatten()[:,None]
    Y_test = YY_test.flatten()[:,None]
    
    xy_test = hstack((X_test, Y_test))
    T_test  = t_test[:, None].repeat(1, xy_test.shape[0])
    xy_test = xy_test.repeat(T_test.shape[0], 1)
    T_test  = T_test.flatten()[:, None]
    
    xyt_test = hstack((xy_test, T_test))
    ##########################################################
    return (xyt_resid, xyt_test)