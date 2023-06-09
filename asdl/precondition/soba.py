from typing import Callable, Tuple, Optional
from contextlib import contextmanager
from math import sqrt
import torch
from torch import Tensor
from torch.optim import Optimizer


ClosureType = Callable[[], Tensor]


class Soba(Optimizer):
    def __init__(
            self, params, lr, data_size: int, mc_samples: int = 1,
            momentum_grad: float = 0.9, momentum_hess: Optional[float] = None,
            weight_decay: float = 0.0, std_init: Optional[float] = None,
            damping: float = 0.0, temperature: float = 1.0,
            clip_radius: float = float("inf")):
        assert lr > 0.0
        assert data_size >= 1
        assert mc_samples >= 1
        assert weight_decay >= 0.0
        assert damping >= 0.0
        assert temperature >= 0
        if momentum_hess is None:
            momentum_hess = 1.0 - lr  # default follows theoretical derivation
        self.mc_samples = mc_samples
        defaults = dict(
            lr=lr, data_size=data_size, mc_samples=mc_samples,
            momentum_grad=momentum_grad, momentum_hess=momentum_hess,
            weight_decay=weight_decay, std_init=std_init, damping=damping,
            temperature=temperature, clip_radius=clip_radius)
        super().__init__(params, defaults)
        self._init_buffers()
        self._reset_samples()            

    def _reset_samples(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    self.state[p]['noise_samples'] = []
                    self.state[p]['grad_samples'] = []

    def _init_buffers(self):
        for group in self.param_groups:
            std_init = group['std_init']
            std_init = 1.0 if std_init is None else std_init
            for p in group['params']:
                if p.requires_grad:
                    self.state[p]['momentum_grad'] = torch.zeros_like(p)
                    self.state[p]['log_std'] = torch.ones_like(
                        p) * torch.log(torch.as_tensor(std_init))

    @contextmanager
    def sampled_params(self, train: bool = False):
        self._sample_weight(train)
        yield
        self._restore_weight_average(train)

    def _sample_weight(self, train):
        for group in self.param_groups:
            rsqrt_n = 1.0 / sqrt(group['data_size'])
            for p in group['params']:
                if p.requires_grad:
                    p_avg = p.data
                    self.state[p]['param_average'] = p_avg
                    normal_sample = torch.randn_like(p)
                    std = torch.exp(self.state[p]['log_std'])
                    p.data = p_avg + rsqrt_n * std * normal_sample
                    if train:  # collect noise sample for training
                        self.state[p]['noise_samples'].append(normal_sample)

    def _restore_weight_average(self, train: bool):
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    p.data = self.state[p]['param_average']
                    self.state[p]['param_average'] = None
                    if train:  # collect grade sample for training
                        self.state[p]['grad_samples'].append(p.grad)

    @torch.no_grad()
    def step(
            self, closure: ClosureType = None
    ) -> Optional[Tensor]:
        if closure is None:
            loss = None
        else:
            losses = []
            for _ in range(self.mc_samples):
                with torch.enable_grad():
                    loss = closure()
                losses.append(loss)
            loss = sum(losses) / self.mc_samples
        self._update()
        self._reset_samples()
        return loss

    def _update(self):
        for group in self.param_groups:
            lr = group['lr']
            lamb = group['weight_decay']
            n = group['data_size']
            d = group['damping']
            m = group['momentum_grad']
            h = group['momentum_hess']
            t = group['temperature']
            clip_radius = group['clip_radius']
            for p in group['params']:
                if p.requires_grad:
                    log_std = self.state[p]['log_std']
                    m_grad = self._update_momentum_grads(p, lamb, m)
                    f = self._compute_f(p, log_std, lamb, n, d, t)
                    self._update_param_averages(
                        p, log_std, lr, clip_radius, m_grad)
                    self.update_log_std_buffers(p, h, f, log_std)

    def _update_momentum_grads(self, p, lamb, m):
        grad_avg = torch.mean(torch.stack(
            self.state[p]['grad_samples'], dim=0), dim=0)
        m_grad = m * self.state[p]['momentum_grad'] + \
            (1 - m) * (lamb * p + grad_avg)
        self.state[p]['momentum_grad'] = m_grad
        return m_grad

    def _compute_f(self, p, log_std, lamb, n, d, t):
        std = torch.exp(log_std)
        temp = [g * eps * std for eps, g in zip(
            self.state[p]['noise_samples'], self.state[p]['grad_samples'])]
        return (lamb + d) * torch.exp(2 * log_std) + sqrt(
            n) * torch.mean(torch.stack(temp, dim=0), dim=0) - t

    def _update_param_averages(
            self, p, log_std, lr, clip_radius, m_grad):
        p.data = p - lr * torch.clip(
            m_grad, min=-clip_radius, max=clip_radius
        ) * torch.exp(2 * log_std)

    def update_log_std_buffers(self, p, h, f, log_std):
        self.state[p]['log_std'] = log_std - 0.5 * torch.log(
            0.5 * (1 + (1 + (1 - h) * f) ** 2))
