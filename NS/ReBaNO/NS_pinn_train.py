import torch
import numpy as np
from timeit import default_timer
torch.set_default_dtype(torch.float)    

def pinn_train(pinn, nu, xt_resid, IC_xt, BC_xt_bottom, BC_xt_top, BC_xt_left, BC_xt_right, epochs_pinn,
               lr_pinn, tol_adam, lr_lbfgs, epochs_lbfgs, using_data=False, Nt=None, xt_data=None):
    
    losses = [pinn.loss(xt_resid, IC_xt, BC_xt_bottom, BC_xt_top, BC_xt_left, BC_xt_right,
                        using_data=using_data, Nt=Nt, xt_data=xt_data).item()] 
    ep = [0]
    optimizer = torch.optim.Adam(pinn.parameters(), lr=lr_pinn)
    step_size = 5000
    gamma = 0.5
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    print(f"Epoch: 0 | Loss: {losses[0]}")
    for i in range(1, epochs_pinn+1):
        # t1 = default_timer()
        train_time1 = default_timer()
        loss_values = pinn.loss(xt_resid, IC_xt, BC_xt_bottom, BC_xt_top, BC_xt_left, BC_xt_right,
                                using_data=using_data, Nt=Nt, xt_data=xt_data) 
        
        if (loss_values.item() < tol_adam):
            losses.append(loss_values.item())
            ep.append(i)
            print(f'Epoch: {i} | Loss: {loss_values.item()} (Stopping Criteria Met)')
            print('PINN Adam Training Completed!\n')
            break
        
        
        optimizer.zero_grad()
        loss_values.backward()
        optimizer.step()
        # scheduler.step()
        train_time2 = default_timer()
        
        # print("Training time for one epoch : ", t2 - t1)
        if (i % 100 == 0) or (i == epochs_pinn):
            losses.append(loss_values.item())
            ep.append(i)
            if (i % 1000 == 0) or (i == epochs_pinn):
                print(f'Epoch: {i} | loss: {loss_values.item()} ep time: {train_time2 - train_time1}')
                
                if (i == epochs_pinn):
                    # print(f'lossIC: {PINN.lossIC(IC_xt).item()}, lossR: {PINN.lossR(xt_resid, f_hat).item()}')
                    print("PINN Adam Training Completed!\n")
        scheduler.step()  
    
    """    
    # L-BFGS training
    print('Begin L-BFGS training!')
    
    optim_lbfgs = torch.optim.LBFGS(list(pinn.parameters()), lr=lr_lbfgs, max_iter=20, 
                                    max_eval=25, tolerance_grad=1e-7)
    
    def closure():
        optim_lbfgs.zero_grad()
        
        loss = pinn.loss(xt_resid, IC_xt, BC_xt_bottom, BC_xt_top, BC_xt_left, BC_xt_right, 
                         using_data=using_data, Nt=Nt, xt_data=xt_data)
        
        losses.append(loss.item())
        # print(f'loss: {loss}')
        loss.backward()
        
        return loss
    
    for ep in range(1, epochs_lbfgs+1):
        
        optim_lbfgs.step(closure)
        
        # with torch.no_grad():
        #    sa_loss, loss = sa_pinn.loss(xt_resid, xt_IC, xt_BC_bottom, xt_BC_top,
        #                          weights_pde, weights_IC)
                
        if np.isnan(losses[-1]) or np.isinf(losses[-1]):
            print("Nan or Inf error in loss! ")
            break
        
        
        if ep % 100 == 0:
            print(f"Epoch: {ep} | loss: {losses[-1]}")
        
            if ep == epochs_lbfgs:
                print('PINN L-BFGS training completed!\n')        
    """  
    return losses