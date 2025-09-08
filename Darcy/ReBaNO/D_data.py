from torch import tensor, linspace, meshgrid, hstack, ones, vstack
import torch

def create_BC_data(Xi, Xf, Yi, Yf, BC_pts):
    ##########################################################
    x_BC = linspace(Xi, Xf, BC_pts+1)
    y_BC = linspace(Yi, Yf, BC_pts+1)
    
    X_BC, Y_BC = meshgrid(x_BC, y_BC, indexing='ij')

    ##########################################################
    BC_left_x = X_BC[0,:][:,None] 
    BC_left_y = Y_BC[0,:][:,None] 
    
    BC_left_xy  = hstack((BC_left_x, BC_left_y)) 
    ##########################################################
    BC_right_x = X_BC[-1,:][:,None] 
    BC_right_y = Y_BC[-1,:][:,None] 
    
    BC_right_xy  = hstack((BC_right_x, BC_right_y)) 
    ##########################################################
    BC_bottom_x = X_BC[:,0][:,None] 
    BC_bottom_y = Y_BC[:,0][:,None] 
    
    BC_bottom_xy  = hstack((BC_bottom_x, BC_bottom_y)) 
    ##########################################################
    BC_top_x = X_BC[:,-1][:,None] 
    BC_top_y = Y_BC[:,-1][:,None] 
    
    BC_top_xy  = hstack((BC_top_x, BC_top_y)) 
    ##########################################################
    return (BC_bottom_xy, BC_top_xy, BC_left_xy, BC_right_xy)  

def create_residual_data(Xi, Xf, Yi, Yf, Nc, N_test):
    ##########################################################
    x_resid = linspace(Xi, Xf, Nc+1)
    y_resid = linspace(Yi, Yf, Nc+1)
    
    XX_resid, YY_resid = meshgrid((x_resid, y_resid), indexing='ij')
    
    X_resid = XX_resid.flatten()[:,None]
    Y_resid = YY_resid.flatten()[:,None]
    
    xy_resid = hstack((X_resid, Y_resid))
    # f_hat_train = ones((xy_resid.shape[0], 1))
    ##########################################################
    x_test = linspace(Xi, Xf, N_test+1)
    y_test = linspace(Xi, Xf, N_test+1)
    
    XX_test, YY_test = meshgrid((x_test, y_test), indexing='ij')
    
    X_test = XX_test.flatten()[:,None]
    Y_test = YY_test.flatten()[:,None]
    
    xy_test = hstack((X_test, Y_test))
    ##########################################################
    return (xy_resid, xy_test)