from contextlib import contextmanager, nullcontext

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import BatchSampler, Subset, DataLoader
from torch.cuda import nvtx

torch_function_class = F.cross_entropy.__class__

_REQUIRES_GRAD_ATTR = '_original_requires_grad'

__all__ = [
    'original_requires_grad', 'record_original_requires_grad',
    'restore_original_requires_grad', 'skip_param_grad', 'im2col_2d',
    'im2col_2d_slow', 'cholesky_inv', 'cholesky_solve', 'smw_inv',
    'PseudoBatchLoaderGenerator', 'nvtx_range', 'has_reduction'
]


def original_requires_grad(module=None, param_name=None, param=None):
    if param is None:
        if module is None or param_name is None:
            raise ValueError('Both module and param_name have to be set.')
        param = getattr(module, param_name, None)
    return param is not None and getattr(param, _REQUIRES_GRAD_ATTR)


def record_original_requires_grad(param):
    setattr(param, _REQUIRES_GRAD_ATTR, param.requires_grad)


def restore_original_requires_grad(param):
    param.requires_grad = getattr(param, _REQUIRES_GRAD_ATTR,
                                  param.requires_grad)


@contextmanager
def skip_param_grad(model, disable=False):
    if not disable:
        for param in model.parameters():
            record_original_requires_grad(param)
            param.requires_grad = False

    yield
    if not disable:
        for param in model.parameters():
            restore_original_requires_grad(param)


def im2col_2d(x: torch.Tensor, conv2d: nn.Module):
    if x.ndim != 4:  # n x c x h_in x w_in
        raise ValueError(f'x.ndim has to be 4. Got {x.ndim}.')
    if not isinstance(conv2d, (nn.Conv2d, nn.ConvTranspose2d)):
        raise TypeError(f'conv2d has to be {nn.Conv2d} or {nn.ConvTranspose2d}. Got {type(conv2d)}.')
    if conv2d.dilation != (1, 1):
        raise ValueError(f'conv2d.dilation has to be (1, 1). Got {conv2d.dilation}.')

    ph, pw = conv2d.padding if conv2d.padding != 'valid' else (0, 0)
    kh, kw = conv2d.kernel_size
    sy, sx = conv2d.stride
    if ph + pw > 0:
        x = F.pad(x, (pw, pw, ph, ph)).data
    x = x.unfold(2, kh, sy)  # n x c x h_out x w_in x kh
    x = x.unfold(3, kw, sx)  # n x c x h_out x w_out x kh x kw
    x = x.permute(0, 1, 4, 5, 2,
                  3).contiguous()  # n x c x kh x kw x h_out x w_out
    x = x.view(x.size(0),
               x.size(1) * x.size(2) * x.size(3),
               x.size(4) * x.size(5))  # n x c(kh)(kw) x (h_out)(w_out)
    return x


def im2col_2d_slow(x: torch.Tensor, conv2d: nn.Module):
    if x.ndim != 4:  # n x c x h_in x w_in
        raise ValueError(f'x.ndim has to be 4. Got {x.ndim}.')
    if not isinstance(conv2d, (nn.Conv2d, nn.ConvTranspose2d)):
        raise TypeError(f'conv2d has to be {nn.Conv2d} or {nn.ConvTranspose2d}. Got {type(conv2d)}.')

    padding = conv2d.padding if conv2d.padding != 'valid' else (0, 0)
    # n x c(k_h)(k_w) x (h_out)(w_out)
    Mx = F.unfold(x,
                  conv2d.kernel_size,
                  dilation=conv2d.dilation,
                  padding=padding,
                  stride=conv2d.stride)

    return Mx


def cholesky_inv(X, damping=1e-7):
    diag = torch.diagonal(X)
    diag += damping
    u = torch.linalg.cholesky(X)
    diag -= damping
    return torch.cholesky_inverse(u)


def cholesky_solve(X, b, damping=1e-7):
    diag = torch.diagonal(X)
    diag += damping
    u = torch.linalg.cholesky(X)
    diag -= damping
    return torch.cholesky_solve(b, u)


def smw_inv(x, damping=1e-7):
    n, d = x.shape  # n x d
    I = torch.eye(d, device=x.device)
    G = x @ x.T  # n x n (Gram matrix)
    diag = torch.diagonal(G)
    diag += damping * n
    Ginv_x = torch.linalg.solve(G, x)  # n x d
    xt_Ginv_x = x.T @ Ginv_x  # d x d
    return (I - xt_Ginv_x) / damping  # d x d


@torch.no_grad()
def ComputePower(mat_g,
                 p,
                 iter_count=100,
                 error_tolerance=1e-6,
                 ridge_epsilon=1e-6):
    """A method to compute G^{-1/p} using a coupled Newton iteration.

  See for example equation 3.2 on page 9 of:
  A Schur-Newton Method for the Matrix p-th Root and its Inverse
  by Chun-Hua Guo and Nicholas J. Higham
  SIAM Journal on Matrix Analysis and Applications,
  2006, Vol. 28, No. 3 : pp. 788-804
  https://pdfs.semanticscholar.org/0abe/7f77433cf5908bfe2b79aa91af881da83858.pdf

  Args:
    mat_g: A square positive semidefinite matrix
    p: a positive integer
    iter_count: Stop iterating after this many rounds.
    error_tolerance: Threshold for stopping iteration
    ridge_epsilon: We add this times I to G, to make is positive definite.
                   For scaling, we multiply it by the largest eigenvalue of G.
  Returns:
    (mat_g + rI)^{-1/p} (r = ridge_epsilon * max_eigenvalue of mat_g).
  """
    shape = list(mat_g.shape)
    if len(shape) == 1:
        return torch.pow(mat_g + ridge_epsilon, -1 / p)
    identity = torch.eye(shape[0], device=mat_g.device)
    if shape[0] == 1:
        return identity
    alpha = -1.0 / p
    max_ev, _, _ = PowerIter(mat_g)
    ridge_epsilon *= max_ev
    mat_g += ridge_epsilon * identity
    z = (1 + p) / (2 * torch.norm(mat_g))
    # The best value for z is
    # (1 + p) * (c_max^{1/p} - c_min^{1/p}) /
    #            (c_max^{1+1/p} - c_min^{1+1/p})
    # where c_max and c_min are the largest and smallest singular values of
    # mat_g.
    # The above estimate assumes that c_max > c_min * 2^p
    # Can replace above line by the one below, but it is less accurate,
    # hence needs more iterations to converge.
    # z = (1 + p) / tf.trace(mat_g)
    # If we want the method to always converge, use z = 1 / norm(mat_g)
    # or z = 1 / tf.trace(mat_g), but these can result in many
    # extra iterations.

    mat_root = identity * torch.pow(z, 1.0 / p)
    mat_m = mat_g * z
    error = torch.max(torch.abs(mat_m - identity))
    count = 0
    while error > error_tolerance and count < iter_count:
        tmp_mat_m = (1 - alpha) * identity + alpha * mat_m
        new_mat_root = torch.matmul(mat_root, tmp_mat_m)
        mat_m = torch.matmul(MatPower(tmp_mat_m, p), mat_m)
        new_error = torch.max(torch.abs(mat_m - identity))
        if new_error > error * 1.2:
            break
        mat_root = new_mat_root
        error = new_error
        count += 1
    return mat_root


@torch.no_grad()
def PowerIter(mat_g, error_tolerance=1e-6, num_iters=100):
    """Power iteration.

  Compute the maximum eigenvalue of mat, for scaling.
  v is a random vector with values in (-1, 1)

  Args:
    mat_g: the symmetric PSD matrix.
    error_tolerance: Iterative exit condition.
    num_iters: Number of iterations.

  Returns:
    eigen vector, eigen value, num_iters
  """
    v = torch.rand(list(mat_g.shape)[0], device=mat_g.device) * 2 - 1
    error = 1
    iters = 0
    singular_val = 0
    while error > error_tolerance and iters < num_iters:
        v = v / torch.norm(v)
        mat_v = torch.mv(mat_g, v)
        s_v = torch.dot(v, mat_v)
        error = torch.abs(s_v - singular_val)
        v = mat_v
        singular_val = s_v
        iters += 1
    return singular_val, v / torch.norm(v), iters


@torch.no_grad()
def MatPower(mat_m, p):
    """Computes mat_m^p, for p a positive integer.

  Args:
    mat_m: a square matrix
    p: a positive integer

  Returns:
    mat_m^p
  """
    if p in [1, 2, 4, 8, 16, 32]:
        p_done = 1
        res = mat_m
        while p_done < p:
            res = torch.matmul(res, res)
            p_done *= 2
        return res

    power = None
    while p > 0:
        if p % 2 == 1:
            power = torch.matmul(mat_m, power) if power is not None else mat_m
        p //= 2
        mat_m = torch.matmul(mat_m, mat_m)
    return power


class PseudoBatchLoaderGenerator:
    """
    Example::
    >>> # create a base dataloader
    >>> dataset_size = 10
    >>> x_all = torch.tensor(range(dataset_size))
    >>> dataset = torch.utils.data.TensorDataset(x_all)
    >>> data_loader = torch.utils.data.DataLoader(dataset, shuffle=True)
    >>>
    >>> # create a pseudo-batch loader generator
    >>> pb_loader_generator = PseudoBatchLoaderGenerator(data_loader, 5)
    >>>
    >>> for i, pb_loader in enumerate(pb_loader_generator):
    >>>     print(f'pseudo-batch at step {i}')
    >>>     print(list(pb_loader))

    Outputs:
    ```
    pseudo-batch at step 0
    [[tensor([0])], [tensor([1])], [tensor([3])], [tensor([6])], [tensor([7])]]
    pseudo-batch at step 1
    [[tensor([8])], [tensor([5])], [tensor([4])], [tensor([2])], [tensor([9])]]
    ```
    """
    def __init__(self,
                 base_data_loader,
                 pseudo_batch_size,
                 batch_size=None,
                 drop_last=None):
        if batch_size is None:
            batch_size = base_data_loader.batch_size
        if pseudo_batch_size % batch_size != 0:
            raise ValueError(f'pseudo_batch_size ({pseudo_batch_size}) '
                             f'needs to be divisible by batch_size ({batch_size})')
        if drop_last is None:
            drop_last = base_data_loader.drop_last
        base_dataset = base_data_loader.dataset
        sampler_cls = base_data_loader.sampler.__class__
        pseudo_batch_sampler = BatchSampler(sampler_cls(
            range(len(base_dataset))),
                                            batch_size=pseudo_batch_size,
                                            drop_last=drop_last)
        self.batch_size = batch_size
        self.pseudo_batch_sampler = pseudo_batch_sampler
        self.base_dataset = base_dataset
        self.base_data_loader = base_data_loader

    def __iter__(self):
        loader = self.base_data_loader
        for indices in self.pseudo_batch_sampler:
            subset_in_pseudo_batch = Subset(self.base_dataset, indices)
            data_loader = DataLoader(
                subset_in_pseudo_batch,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=loader.num_workers,
                collate_fn=loader.collate_fn,
                pin_memory=loader.pin_memory,
                drop_last=False,
                timeout=loader.timeout,
                worker_init_fn=loader.worker_init_fn,
                multiprocessing_context=loader.multiprocessing_context,
                generator=loader.generator,
                prefetch_factor=loader.prefetch_factor,
                persistent_workers=loader.persistent_workers)
            yield data_loader

    def __len__(self) -> int:
        return len(self.pseudo_batch_sampler)


@contextmanager
def nvtx_range(msg, *args, **kwargs):
    if torch.cuda.is_available():
        yield nvtx.range(msg, *args, **kwargs)
    else:
        yield nullcontext()


def has_reduction(func):
    if isinstance(func, nn.Module):
        return hasattr(func, 'reduction')
    elif isinstance(func, torch_function_class):
        return 'reduction' in func.__code__.co_varnames
    return False


