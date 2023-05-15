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
        self.beta2 = 1 - config.ema_decay

    @torch.no_grad()
    def update_curvature(self):
        """
        Performs a single optimization step.
        """
        for name, module in self.module_dict.items():
            if module.weight.grad is not None:
                self._update_curvature_module(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                self._update_curvature_module(module.bias)

    def _update_curvature_module(self, module_weight):
        grad = module_weight.grad
        exp_avg = getattr(module_weight, 'exp_avg', None)
        if exp_avg is None:
            exp_avg = torch.zeros_like(module_weight)
            setattr(module_weight, 'exp_avg', exp_avg)
        exp_avg_sq = exp_avg.mul_(self.beta2) + (1 - self.beta2) * (grad * grad)

        bias_correction2 = 1 / (1 - self.beta2 ** (self.state['step'] + 1))
        exp_avg_sq_hat = exp_avg_sq * bias_correction2
                
        setattr(module_weight, 'exp_avg', exp_avg_sq)
        denom = exp_avg_sq_hat.sqrt() + self.eps
        setattr(module_weight, 'curvature', denom)

    @torch.no_grad()
    def precondition(self):
        """
        Performs a single optimization step.
        """
        loss = None
        for name, module in self.module_dict.items():
            if module.weight.grad is not None:
                self._precondition_module(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                self._precondition_module(module.bias)
        return loss

    def _precondition_module(self, module_weight):
        grad = module_weight.grad
        denom = getattr(module_weight, 'curvature', None)
        if denom is not None:
            module_weight.grad.data = grad.clone().detach() / denom.clone().detach()
