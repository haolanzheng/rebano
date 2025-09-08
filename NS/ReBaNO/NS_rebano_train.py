import torch

torch.set_default_dtype(torch.float)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def rebano_train(rebano, nu, xt_resid, IC_xt, BC_xt_bottom, BC_xt_top, BC_xt_left, BC_xt_right, 
              epochs_rebano, lr_rebano, arg=None, largest_loss=None, largest_case=None, testing=False):
     
    # GD = grad_descent(nu, xt_resid, IC_xt, BC_xt, u0, P_resid_values, P_IC_values, 
    #                 P_BC_values, P_x_term, Pt_nu_P_xx_term, lr_rebano)
    optimizer = torch.optim.Adam(rebano.parameters(), lr=lr_rebano)
    if (testing == False): 
        
        loss_values = rebano.loss()
        # lossR = rebano.lossR()
        # lossBC = rebano.lossBC()
        # lossIC = rebano.lossIC()
        # print(lossR, lossBC, lossIC, loss_values)
        
        # loss_R = rebano.lossR()
        # loss_IC = rebano.lossICBC(datatype='initial')
        # loss_BC = rebano.lossICBC(datatype='boundary')
        # print(loss_values, loss_R, loss_IC, loss_BC)
        # rebano.train()
        for i in range(1, epochs_rebano+1):
            
                
            if (loss_values < largest_loss): 
                break
                
            else:
                
                
                # rebano.linears[1].weight.data = GD.update(c)
                optimizer.zero_grad()
                loss_values.backward()
                optimizer.step()
                
                
                # if (i%200 == 0):
                 #    print(c)
                
                # if (i%1000 == 0):
                    # print('Epoches: ', i, 'ReBaNO loss: ', rebano.loss().item())
                
                if (i == epochs_rebano):
                    # c = rebano.linears[-1].weight.data
                    loss_R = rebano.lossR()
                    # loss_BC = rebano.lossBC(BC_xt_bottom, BC_xt_top)
                    # loss_R  = rebano.lossR(xt_resid)
                    # print(f"lossR: {loss_R}, lossBC: {loss_BC}")
                    loss = rebano.loss()
                    # lossIC = rebano.lossICBC(datatype='initial')
                    # lossBC = rebano.lossICBC(datatype='boundary')
                    # print(f'lossIC: {lossIC.item()}, lossBC: {lossBC.item()}')
                    # print("c: ", c)
                    
                    largest_case = arg
                    largest_loss = loss 
                    
                    print(f"arg: {arg}, residual loss: {loss_R.item()}, total loss: {loss.item()}")
                    
                    
            loss_values = rebano.loss()
        return largest_loss, largest_case
    
    elif (testing):
        for i in range(1, epochs_rebano+1):
            loss_values = rebano.loss()
            # c = rebano.linears[1].weight.data.view(-1)
            # rebano.linears[1].weight.data = GD.update(c)
            optimizer.zero_grad()
            loss_values.backward()
            optimizer.step()
        
        return rebano.loss().item()
            