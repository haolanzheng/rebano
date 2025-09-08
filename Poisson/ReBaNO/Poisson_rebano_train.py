import torch
from torch import nn

torch.set_default_dtype(torch.float)


def rebano_train(rebano, x_resid, BC_x, epochs_rebano, lr_rebano, ind=None, 
              largest_loss=None, largest_case=None, testing=False, strong_form=False, u_data=None):

    
    optimizer = torch.optim.LBFGS(rebano.parameters(), lr=lr_rebano)
    
    def closure():
        optimizer.zero_grad()
        
        loss_values = rebano.loss()
        loss_values.backward()
        
        return loss_values
    
    if testing == False:  # Need to comp. loss for training
        if strong_form == False:
            loss_values = rebano.loss()

            for i in range(1, epochs_rebano + 1):
                # print(ind, 'case loss_values =', loss_values.item())
                if loss_values < largest_loss:
                    break

                else:
                    
                    optimizer.step(closure)

                    if i == epochs_rebano:
                        # print(rebano.lossR().item(), rebano.lossBC().item())
                        # print(ind)
                        largest_case = ind
                        largest_loss = rebano.loss()

                loss_values = rebano.loss()
                
        elif u_data is not None:
            loss_values = rebano.lossD(u_data)
            
            for i in range(1, epochs_rebano + 1):
                # print(ind, 'case loss_values =', loss_values.item())
                if loss_values < largest_loss:
                    break

                else:
                    optimizer.step(closure)

                    if i == epochs_rebano:
                        c = rebano.linears[-1].weight.data.cpu().numpy()
                        print(c)
                        print(f'largest loss : {rebano.lossD(u_data).item()}, arg max : {ind}')
                        largest_case = ind
                        largest_loss = rebano.lossD(u_data)

                loss_values = rebano.lossD(u_data)

        return largest_loss, largest_case

    elif testing:
        for i in range(1, epochs_rebano + 1):  # Don't need to comp. loss for testing
            
            # c = rebano.linears[1].weight.data.view(-1)
            # rebano.linears[1].weight.data = GD.update(c)

            optimizer.step(closure)
        
        return rebano.loss().item()