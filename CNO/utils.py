import torch

def count_params(net):
    count = 0
    for p in net.parameters():
        count += p.numel()
    return count

def save_ckpt(path, model, optimizer=None, scheduler=None):
    model_state = model.state_dict()
    if optimizer:
        optim_state = optimizer.state_dict()
    else:
        optim_state = None
    
    if scheduler:
        scheduler_state = scheduler.state_dict()
    else:
        scheduler_state = None
    torch.save({
        'model': model_state, 
        'optim': optim_state, 
        'scheduler': scheduler_state
    }, path)
    print(f'Checkpoint is saved to {path}')

class LpLoss(object):
    '''
    loss function with rel/abs Lp loss
    '''
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        # h = 1.0 / (x.size()[1] - 1.0)

        all_norms = torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)