import torch
import math
from .prec_grad_maker import PreconditionedGradientMaker, PreconditioningConfig
import torch.nn as nn

class AdamGradientMaker(PreconditionedGradientMaker):
    """
    implements ADAM Algorithm, as a preceding step.
    """
    def __init__(self, model: nn.Module, config):
        super().__init__(model, config)
        self.eps = config.damping
        self.momentum = config.momentum
        self.beta2 = 1-config.ema_decay

    @torch.no_grad()
    def update_curvature(self):
        """
        Performs a single optimization step.
        """
        loss = None
        for (name, module) in self.module_dict.items():
            if module.weight.grad is not None:
                grad = module.weight.grad
                if not hasattr(module.weight,'exp_avg'):
                    module.weight.exp_avg = torch.zeros_like(module.weight.grad)
                
                bias_correction1 = 1 / (1 - self.momentum ** (self.state['step']+1))
                exp_avg_sq = module.weight.exp_avg
                # RMS
                exp_avg_sq = torch.mul(exp_avg_sq, self.beta2) + (1-self.beta2)*(grad*grad)
                bias_correction2 = 1 / (1 - self.beta2 ** (self.state['step']+1))
                module.weight.exp_avg = exp_avg_sq * bias_correction2
            if hasattr(module,'bias') and hasattr(module.bias,'grad') and module.bias.grad is not None:
                grad = module.bias.grad
                if not hasattr(module.bias,'exp_avg'):
                    module.bias.exp_avg = torch.zeros_like(module.bias.grad)
                exp_avg_sq = module.bias.exp_avg
                # RMS
                exp_avg_sq = torch.mul(exp_avg_sq, self.beta2) + (1-self.beta2)*(grad*grad)
                module.bias.exp_avg = exp_avg_sq
                denom = exp_avg_sq.sqrt() + self.eps
                module.bias.curvature = denom 

    @torch.no_grad()
    def precondition(self):
        """
        Performs a single optimization step.
        """
        loss = None
        for (name, module) in self.module_dict.items():
            if module.weight.grad is not None:
                grad = module.weight.grad
                denom = module.weight.exp_avg.sqrt() + self.eps
                module.weight.grad = grad / denom
            if hasattr(module,'bias') and hasattr(module.bias,'grad') and module.bias.grad is not None:
                grad = module.bias.grad
                denom = module.bias.exp_avg.sqrt() + self.eps
                module.bias.grad = grad / denom  
        return loss
    