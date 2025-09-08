import torch
from time import perf_counter
torch.set_default_dtype(torch.float)

def pinn_train(PINN, x_resid, BC_x, BC_u, epochs_pinn, lr_pinn, tol_pinn):
    losses = [PINN.loss(x_resid, BC_x, BC_u).item()]

    optimizer = torch.optim.Adam(PINN.parameters(), lr=lr_pinn)
    gamma = 0.5
    step_size = 5000
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    # print(f"Epoch: 0 | Loss: {losses[0]}")
    for i in range(1, epochs_pinn+1):
        loss_values = PINN.loss(x_resid, BC_x, BC_u) 
        if loss_values < tol_pinn:
            print(f'Epoch: {i} | loss: {loss_values.item()} (Stopping criteria met!)')
            print("PINN training completed!")
            break
        t1 = perf_counter()
        optimizer.zero_grad()
        loss_values.backward()
        optimizer.step()
        t2 = perf_counter()
        if (i % 5000 == 0) or (i == epochs_pinn):
            losses.append(loss_values.item())
            print(f'Epoch: {i} | loss: {loss_values.item()}\t ep time: {t2 - t1}')
    
            if (i == epochs_pinn):
                print("PINN Training Completed\n")
        scheduler.step()   
    return losses