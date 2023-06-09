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
OPTIM_NGD_UNIT_WISE = 'ngd_unitwise'
OPTIM_NGD_FULL = 'ngd_full'
OPTIM_NGD_DIAG = 'ngd_diag'
OPTIM_LARS = 'lars'
OPTIM_ADAM_ASDL = 'adam_asdl'
OPTIM_ADAM_KFAC = 'adam_kfac'
OPTIM_ADAM_SHAMPOO = 'adam_shampoo'
OPTIM_SHAMPOO_KFAC = 'shampoo_kfac'
OPTIM_ADAM_PSGD = 'adam_psgd'
OPTIM_SOPHIAG = 'sophiag'
OPTIM_SOBA = 'soba'

def create_grad_maker(model,optimizer,args):
    if 'None' in args.ignore_modules:
        args.ignore_modules = []
    args.ignore_modules.extend([nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d,nn.LayerNorm])
    config = PreconditioningConfig(data_size=args.batch_size,
                                    damping=args.damping,
                                    ema_decay = args.ema_decay,
                                    preconditioner_upd_interval=args.curvature_update_interval,
                                    curvature_upd_interval=args.curvature_update_interval,
                                    grad_norm_clip = args.grad_norm_clip,
                                    grad_value_clip = args.grad_value_clip,
                                    after_grad_norm_clip = args.after_grad_norm_clip,
                                    after_grad_value_clip = args.after_grad_value_clip,
                                    inv_exp = args.inv_exp,
                                    dmp_technique = args.dmp_technique,
                                    ignore_modules=args.ignore_modules,
                                    )

    if args.optim == OPTIM_KFAC_MC:
        grad_maker = KfacGradientMaker(model, config)
    elif args.optim == OPTIM_KFAC_EMP or args.optim == OPTIM_ADAM_KFAC or args.optim == OPTIM_SHAMPOO_KFAC:
        grad_maker = KfacEmpGradientMaker(model, config)
    elif args.optim == OPTIM_ADAM_ASDL:
        grad_maker = AdamGradientMaker(model, config)
    elif args.optim == OPTIM_NGD_FULL:
        grad_maker = FullNaturalGradientMaker(model, config)
    elif args.optim == OPTIM_NGD_LAYER_WISE:
        grad_maker = LayerWiseNaturalGradientMaker(model, config)
    elif args.optim == OPTIM_NGD_UNIT_WISE:
        grad_maker = UnitWiseNaturalGradientMaker(model, config)
    elif args.optim == OPTIM_NGD_DIAG:
        grad_maker = DiagNaturalGradientMaker(model, config)
    elif args.optim == OPTIM_SHAMPOO or args.optim == OPTIM_ADAM_SHAMPOO:
        grad_maker = ShampooGradientMaker(model,config,block_size = args.block_size,)
    elif args.optim == OPTIM_SMW_NGD:
        grad_maker = SmwEmpNaturalGradientMaker(model, config)
    elif args.optim == OPTIM_FULL_PSGD:
        grad_maker = PsgdGradientMaker(model)
    elif args.optim == OPTIM_KRON_PSGD or args.optim == OPTIM_ADAM_PSGD:
        grad_maker = KronPsgdGradientMaker(model,config,precond_lr=args.precond_lr)
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
    elif args.optim == OPTIM_ADAM_ASDL:
        grad_maker = AdamGradientMaker(model, config)
    else:
        grad_maker = PreconditionedGradientMaker(model,config)

    return grad_maker