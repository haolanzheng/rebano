import torch
import time

def vpinn_train(vpinn, lr_pinn, epochs_pinn, tol_adam, lr_lbfgs, epochs_lbfgs):
           
    losses = [] 
    ep = []
    optimizer = torch.optim.Adam(vpinn.parameters(), lr=lr_pinn)
    
    step_size = 10000 
    gamma = 0.5
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    wr = 1.0
    wb = 100.0
    for i in range(1, epochs_pinn+1):
        
        train_time1 = time.perf_counter()
        
        lossr, lossb = vpinn.loss()
        loss = lossr + lossb
        wloss = wr * lossr + wb * lossb
        
        if (loss.item() < tol_adam):
            losses.append(loss.item())
            ep.append(i)
            print(f'Epoch: {i} | Loss: {loss.item()} (Stopping Criteria Met)')
            print('VPINN Adam Training Completed!\n')
            break
        
        
        optimizer.zero_grad()
        wloss.backward()
        optimizer.step()
        scheduler.step()
        train_time2 = time.perf_counter()
        
        
        # print("Training time for one epoch : ", t2 - t1)
        if (i % 100 == 0) or (i == epochs_pinn):
            losses.append(loss.item())
            ep.append(i)

            if (i % 1000 == 0) or (i == epochs_pinn):
                print(f'Epoch: {i} | loss : {loss.item()} \t ep time: {train_time2 - train_time1}')
                
                
                if (i == epochs_pinn):
                    # print(f'lossIC: {PINN.lossIC(IC_xt).item()}, lossR: {PINN.lossR(xt_resid, f_hat).item()}')
                    print("VPINN Adam Training Completed!\n")
        # scheduler.step() 
    
    """
    # L-BFGS training
    print('Begin L-BFGS training!')
    
    optim_lbfgs = torch.optim.LBFGS(list(vpinn.parameters()), lr=lr_lbfgs, max_iter=20, 
                                    max_eval=25, tolerance_grad=1e-7)
    
    def closure():
        optim_lbfgs.zero_grad()
        
        loss = vpinn.loss()
        
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
                print('VPINN L-BFGS training completed!\n') 
    """
    return losses