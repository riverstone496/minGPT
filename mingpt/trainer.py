"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import time,math
from collections import defaultdict
import warmup_scheduler

import torch
from torch.utils.data.dataloader import DataLoader
from mingpt.utils import CfgNode as CN

import asdl
from torch.optim.lr_scheduler import CosineAnnealingLR

class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = 'auto'
        # dataloder parameters
        C.num_workers = 4
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0
        return C

    def __init__(self, config, model, train_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer, self.grad_maker = model.configure_optimizers(config)

        if config.scheduler == 'cosine':
            if config.warmup != 0:
                base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.max_iters, eta_min=0)
                self.scheduler = warmup_scheduler.GradualWarmupScheduler(self.optimizer, multiplier=1., total_epoch=config.warmup, after_scheduler=base_scheduler)
            else:
                self.scheduler=CosineAnnealingLR(self.optimizer, T_max=config.max_iters,eta_min=0)
        else:
            self.scheduler = None

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        while True:

            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            # forward the model
            #logits, self.loss = model(x, y)
            #model.zero_grad(set_to_none=True)
            # backprop and update the parameters
            #self.loss.backward()
            
            if config.optim != asdl.OPTIM_SOPHIAG:
                model.zero_grad(set_to_none=True)
                dummy_y = self.grad_maker.setup_model_call(model, x, y)
                self.grad_maker.setup_loss_repr(dummy_y[1])
                logits, self.loss = self.grad_maker.forward_and_backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
            if config.optim == asdl.OPTIM_SOPHIAG:
                logits, loss = model(x, y)
                loss.backward()
                self.optimizer.step(bs=config.batch_size)
                self.optimizer.zero_grad(set_to_none=True)
                # update hessian EMA
                logits, loss = model(x, None)
                samp_dist = torch.distributions.Categorical(logits=logits)
                y_sample = samp_dist.sample()
                loss_sampled = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y_sample.view(-1), ignore_index=-1)
                loss_sampled.backward()
                self.optimizer.update_hessian()
                self.optimizer.zero_grad(set_to_none=True)

            logits, self.loss = model(x, y)

            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break

            if math.isnan(self.loss) or self.loss > 100:
                print('math.isnan(self.loss) or self.loss > 100')
                break

            # eigenlist = []
            # for module in self.grad_maker.module_dict.values():
                
            #     matrix = self.grad_maker._get_module_symmatrix(module, asdl.SHAPE_KRON)
            #     if matrix is None:
            #         continue
            #     A,B=matrix.A,matrix.B

            #     LA = torch.linalg.eigvalsh(A + 1e-8*torch.eye(A.size()[0], A.size()[1]).cuda())
            #     LB = torch.linalg.eigvalsh(B + 1e-8*torch.eye(B.size()[0], B.size()[1]).cuda())
            #     LA.sort(descending=True)
            #     LB.sort(descending=True)

            #     for a in LA[:10]:
            #         for b in LB[:10]:
            #             eigenlist.append(float(a*b))

            # eigenlist.sort(reverse=True)
            # print(self.iter_num,eigenlist[:20])
