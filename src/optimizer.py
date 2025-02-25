import torch
from torch.optim import Optimizer
import numpy as np

class NoisySGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0, dampening=0, weight_decay=0, noise_std=0.01):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if noise_std < 0.0:
            raise ValueError("Invalid noise_std value: {}".format(noise_std))
        
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, noise_std=noise_std)
        super(NoisySGD, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            dampening = group['dampening']
            weight_decay = group['weight_decay']
            noise_std = group['noise_std']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    d_p = buf
                
                # Добавляем шум
                noise = torch.randn_like(d_p) * noise_std
                d_p.add_(noise)
                
                p.data.add_(-lr, d_p)
                
        return loss
