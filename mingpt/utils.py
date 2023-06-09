
import os
import sys
import json
import random
from ast import literal_eval

import numpy as np
import torch
from torch import nn

import wandb

# -----------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_logging(config):
    """ monotonous bookkeeping """
    work_dir = config.system.work_dir
    # create the work directory if it doesn't already exist
    os.makedirs(work_dir, exist_ok=True)
    # log the args (if any)
    with open(os.path.join(work_dir, 'args.txt'), 'w') as f:
        f.write(' '.join(sys.argv))
    # log the config itself
    with open(os.path.join(work_dir, 'config.json'), 'w') as f:
        f.write(json.dumps(config.to_dict(), indent=4))

class CfgNode:
    """ a lightweight configuration class inspired by yacs """
    # TODO: convert to subclass from a dict like in yacs?
    # TODO: implement freezing to prevent shooting of own foot
    # TODO: additional existence/override checks when reading/writing params?

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return self._str_helper(0)

    def _str_helper(self, indent):
        """ need to have a helper to support nested indentation for pretty printing """
        parts = []
        for k, v in self.__dict__.items():
            if isinstance(v, CfgNode):
                parts.append("%s:\n" % k)
                parts.append(v._str_helper(indent + 1))
            else:
                parts.append("%s: %s\n" % (k, v))
        parts = [' ' * (indent * 4) + p for p in parts]
        return "".join(parts)

    def to_dict(self):
        """ return a dict representation of the config """
        return { k: v.to_dict() if isinstance(v, CfgNode) else v for k, v in self.__dict__.items() }

    def merge_from_dict(self, d):
        self.__dict__.update(d)

    def merge_from_args(self, args):
        """
        update the configuration from a list of strings that is expected
        to come from the command line, i.e. sys.argv[1:].

        The arguments are expected to be in the form of `--arg=value`, and
        the arg can use . to denote nested sub-attributes. Example:

        --model.n_layer=10 --trainer.batch_size=32
        """
        for arg in args:

            keyval = arg.split('=')
            assert len(keyval) == 2, "expecting each override arg to be of form --arg=value, got %s" % arg
            key, val = keyval # unpack

            # first translate val into a python object
            try:
                val = literal_eval(val)
                """
                need some explanation here.
                - if val is simply a string, literal_eval will throw a ValueError
                - if val represents a thing (like an 3, 3.14, [1,2,3], False, None, etc.) it will get created
                """
            except ValueError:
                pass

            # find the appropriate object to insert the attribute into
            assert key[:2] == '--'
            key = key[2:] # strip the '--'
            keys = key.split('.')
            obj = self
            for k in keys[:-1]:
                obj = getattr(obj, k)
            leaf_key = keys[-1]

            # ensure that this attribute exists
            assert hasattr(obj, leaf_key), f"{key} is not an attribute that exists in the config"

            # overwrite the attribute
            print("command line overwriting config attribute %s with %s" % (key, val))
            setattr(obj, leaf_key, val)

def make_config(config, parser):
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--beta2', type=float, default=0.95)
    
    parser.add_argument('--grad_norm_clip', type=float, default=1.0)
    parser.add_argument('--grad_value_clip', type=float, default=-1)
    parser.add_argument('--after_grad_norm_clip', type=float, default=-1.0)
    parser.add_argument('--after_grad_value_clip', type=float, default=-1.0)

    parser.add_argument('--datapath', type=str, default='./data/enwik8')
    parser.add_argument('--model_type', type=str, default='gpt-mini')
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--max_iters', type=int, default=1e+4)
    parser.add_argument('--warmup', type=int, default=0)

    parser.add_argument('--inv_exp', type=float, default=-1)
    parser.add_argument('--shampoo_damping', type=float, default=1e-8)
    parser.add_argument('--dmp_technique', type=str, default='heuristics')

    parser.add_argument('--ndigit', type=int, default=2)
    parser.add_argument('--precond_lr', type=float, default=0.01)

    parser.add_argument('--optim', default='adam_asdl')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--ignore_modules', type=str, default='None')

    parser.add_argument('--block_size', type=int, default=128)
    parser.add_argument('--scheduler', type=str, default='cosine')

    parser.add_argument('--curvature_update_interval', type=int, default=1)
    parser.add_argument('--damping', type=float, default=0.000001)
    parser.add_argument('--ema_decay', type=float, default=0.9)
    parser.add_argument('--rho', type=float, default=0.04)

    parser.add_argument('--wandb', action='store_false', default=True)

    args = parser.parse_args()
    if args.wandb:
        wandb.init( config=vars(args).copy(),
                    entity=os.environ.get('WANDB_ENTITY', None),
                    project=os.environ.get('WANDB_PROJECT', None),
                    )

    args.ignore_modules = args.ignore_modules.split(',')
    
    config.trainer.batch_size = args.batch_size
    config.trainer.learning_rate = args.learning_rate
    config.trainer.weight_decay = args.weight_decay
    config.trainer.betas = (args.momentum, args.beta2)

    config.trainer.grad_norm_clip = args.grad_norm_clip
    config.trainer.grad_value_clip = args.grad_value_clip
    config.trainer.after_grad_norm_clip = args.after_grad_norm_clip
    config.trainer.after_grad_value_clip = args.after_grad_value_clip

    config.trainer.num_workers = args.num_workers
    config.trainer.optim = args.optim
    config.trainer.ignore_modules = args.ignore_modules
    config.trainer.precond_lr = args.precond_lr
    config.trainer.rho = args.rho
    config.trainer.warmup = args.warmup

    config.trainer.momentum = args.momentum
    config.trainer.curvature_update_interval = args.curvature_update_interval
    config.trainer.damping = args.damping
    config.trainer.ema_decay = args.ema_decay
    config.trainer.max_iters=args.max_iters
    config.trainer.block_size = args.block_size

    config.trainer.inv_exp = args.inv_exp
    config.trainer.shampoo_damping = args.shampoo_damping
    config.trainer.dmp_technique = args.dmp_technique

    config.trainer.scheduler = args.scheduler

    config.model.model_type = args.model_type
    config.data.datapath = args.datapath
    config.data.wandb = args.wandb
    config.data.ndigit = args.ndigit

    return config