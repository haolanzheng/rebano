import torch
from time import perf_counter

torch.set_default_dtype(torch.float)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def rebano_train(ReBaNO, epochs_rebano, lr_rebano, arg=None, largest_loss=None, largest_case=None, testing=False):
     
    # GD = grad_descent(nu, xt_resid, IC_xt, BC_xt, u0, P_resid_values, P_IC_values, 
    #                 P_BC_values, P_x_term, Pt_nu_P_xx_term, lr_rebano)
    # optimizer = torch.optim.Adam(ReBaNO.parameters(), lr=lr_rebano)
    optimizer = torch.optim.LBFGS(list(ReBaNO.parameters()), lr=lr_rebano, max_iter=20, 
                                   max_eval=25, tolerance_grad=1e-7)
    wr = 1.0
    wb = 1.0
    def closure():
        optimizer.zero_grad()
        
        lossr, lossb = ReBaNO.loss() 
        loss = wr * lossr + wb * lossb
        loss.backward()
        
        return loss
    
    if (testing == False): 
        
        for i in range(1, epochs_rebano+1):
            lossr, lossb = ReBaNO.loss()
            loss_values = wr * lossr + wb * lossb
            
            if (loss_values < largest_loss): 
                break
                
            else:
                
                optimizer.step(closure)
                
                if (i == epochs_rebano):
                    # c = ReBaNO.linears[-1].weight.data

                    lossr, lossb = ReBaNO.loss()
                    loss = lossr + lossb
                    largest_case = arg
                    largest_loss = loss 
                    
                    print(f"arg: {arg}, loss: {loss.item()}")
                    

        return largest_loss, largest_case
    
    elif (testing):
        for i in range(1, epochs_rebano+1):
            lossr, lossb = ReBaNO.loss()
            loss_values = lossr + lossb
            
            optimizer.step(closure)
        
        return loss_values.item()
            