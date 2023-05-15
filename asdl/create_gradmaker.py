import torch
from torch import nn
import torch_optimizer as optim
from .precondition import *

OPTIM_SGD = 'sgd'
OPTIM_ADAM = 'adam'
OPTIM_ADAMW = 'adamw'
OPTIM_SHAMPOO='shampoo'
OPTIM_KFAC_MC = 'kfac_mc'
OPTIM_KFAC_EMP = 'kfac_emp'
OPTIM_NOISY_KFAC_MC = 'noisy_kfac_mc'
OPTIM_SMW_NGD = 'smw_ngd'
OPTIM_FULL_PSGD = 'full_psgd'
OPTIM_KRON_PSGD = 'psgd'
OPTIM_NEWTON = 'newton'
OPTIM_ABS_NEWTON = 'abs_newton'
OPTIM_KBFGS = 'kbfgs'
OPTIM_CURVE_BALL = 'curve_ball'
OPTIM_SENG = 'seng'
OPTIM_ADAHESSIAN = 'adahessian'
OPTIM_SWATS = 'swats'
OPTIM_FOOF = 'foof'
OPTIM_BOOB = 'boob'
OPTIM_NGD_LAYER_WISE = 'ngd_layerwise'
OPTIM_NGD_FULL = 'ngd_full'
OPTIM_LARS = 'lars'
OPTIM_ADAM_ASDL = 'adam_asdl'
OPTIM_ADAM_KFAC = 'adam_kfac'

def create_grad_maker(model,optimizer,args):
    config = PreconditioningConfig(data_size=args.batch_size,
                                    damping=args.damping,
                                    ema_decay = args.ema_decay,
                                    preconditioner_upd_interval=args.curvature_update_interval,
                                    curvature_upd_interval=args.curvature_update_interval,
                                    grad_norm_clip = args.grad_norm_clip,
                                    #ignore_modules=[nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d,nn.LayerNorm],
                                    )

    if args.optim == OPTIM_KFAC_MC:
        grad_maker = KfacGradientMaker(model, config)
    elif args.optim == OPTIM_KFAC_EMP or args.optim == OPTIM_ADAM_KFAC:
        grad_maker = KfacEmpGradientMaker(model, config)
    elif args.optim == OPTIM_ADAM_ASDL:
        grad_maker = AdamGradientMaker(model, config)
    elif args.optim == OPTIM_NGD_FULL:
        grad_maker = FullNaturalGradientMaker(model, config)
    elif args.optim == OPTIM_NGD_LAYER_WISE:
        grad_maker = LayerWiseNaturalGradientMaker(model, config)
    elif args.optim == OPTIM_SHAMPOO:
        grad_maker = ShampooGradientMaker(model,config)
    elif args.optim == OPTIM_SMW_NGD:
        grad_maker = SmwEmpNaturalGradientMaker(model, config)
    elif args.optim == OPTIM_FULL_PSGD:
        grad_maker = PsgdGradientMaker(model)
    elif args.optim == OPTIM_KRON_PSGD:
        grad_maker = KronPsgdGradientMaker(model,config)
    elif args.optim == OPTIM_NEWTON:
        grad_maker = NewtonGradientMaker(model, config)
    elif args.optim == OPTIM_ABS_NEWTON:
        grad_maker = NewtonGradientMaker(model, config)
    elif args.optim == OPTIM_KBFGS:
        grad_maker = KronBfgsGradientMaker(model, config)
    elif args.optim == OPTIM_CURVE_BALL:
        grad_maker = CurveBallGradientMaker(model, config)
    elif args.optim == OPTIM_SENG:
        grad_maker = SengGradientMaker(model,config=config)
    else:
        grad_maker = PreconditionedGradientMaker(model,config)

    return grad_maker