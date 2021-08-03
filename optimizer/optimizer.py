import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from lib.config import cfg
import lr_scheduler
from optimizer.radam import RAdam, AdamW
import math


class Optimizer(nn.Module):
    def __init__(self, model, dataloader):
        super(Optimizer, self).__init__()
        self.setup_optimizer(model, dataloader)

    def setup_optimizer(self, model, dataloader):
        params = []
        for key, value in model.named_parameters():
            if not value.requires_grad:
                continue
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "bias" in key:
                lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR 
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
            if "bert" in key:
                lr = cfg.SOLVER.BASE_LR * 0.1
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        if cfg.SOLVER.TYPE == 'SGD':
            self.optimizer = torch.optim.SGD(
                params, 
                lr = cfg.SOLVER.BASE_LR, 
                momentum = cfg.SOLVER.SGD.MOMENTUM
            )
        elif cfg.SOLVER.TYPE == 'ADAM':
            self.optimizer = torch.optim.Adam(
                params,
                lr = cfg.SOLVER.BASE_LR, 
                betas = cfg.SOLVER.ADAM.BETAS, 
                eps = cfg.SOLVER.ADAM.EPS
            )
        elif cfg.SOLVER.TYPE == 'ADAMAX':
            self.optimizer = torch.optim.Adamax(
                params,
                lr = cfg.SOLVER.BASE_LR, 
                betas = cfg.SOLVER.ADAM.BETAS, 
                eps = cfg.SOLVER.ADAM.EPS
            )
        elif cfg.SOLVER.TYPE == 'ADAGRAD':
            self.optimizer = torch.optim.Adagrad(
                params,
                lr = cfg.SOLVER.BASE_LR
            )
        elif cfg.SOLVER.TYPE == 'RMSPROP':
            self.optimizer = torch.optim.RMSprop(
                params, 
                lr = cfg.SOLVER.BASE_LR
            )
        elif cfg.SOLVER.TYPE == 'RADAM':
            self.optimizer = RAdam(
                params, 
                lr = cfg.SOLVER.BASE_LR, 
                betas = cfg.SOLVER.ADAM.BETAS, 
                eps = cfg.SOLVER.ADAM.EPS
            )
        else:
            raise NotImplementedError

        if cfg.SOLVER.LR_POLICY.TYPE == 'Fix':
            self.scheduler = None
        elif cfg.SOLVER.LR_POLICY.TYPE == 'Step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size = cfg.SOLVER.LR_POLICY.STEP_SIZE, 
                gamma = cfg.SOLVER.LR_POLICY.GAMMA
            )
        elif cfg.SOLVER.LR_POLICY.TYPE == 'Plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,  
                factor = cfg.SOLVER.LR_POLICY.PLATEAU_FACTOR, 
                patience = cfg.SOLVER.LR_POLICY.PLATEAU_PATIENCE
            )
        elif cfg.SOLVER.LR_POLICY.TYPE == 'Noam':
            self.scheduler = lr_scheduler.create(
                'Noam', 
                self.optimizer,
                model_size = cfg.SOLVER.LR_POLICY.MODEL_SIZE,
                factor = cfg.SOLVER.LR_POLICY.FACTOR,
                warmup = cfg.SOLVER.LR_POLICY.WARMUP
            )
        elif cfg.SOLVER.LR_POLICY.TYPE == 'MultiStep':
            self.scheduler = lr_scheduler.create(
                'MultiStep', 
                self.optimizer,
                milestones = cfg.SOLVER.LR_POLICY.STEPS,
                gamma = cfg.SOLVER.LR_POLICY.GAMMA
            )
        elif cfg.SOLVER.LR_POLICY.TYPE == 'WarmupCosine':
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=cfg.SOLVER.LR_POLICY.WARMUP,
                num_training_steps=len(dataloader) * cfg.SOLVER.MAX_EPOCH,
                )
        else:
            raise NotImplementedError

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def scheduler_step(self, lrs_type, val=None):
        if self.scheduler is None:
            return

        if cfg.SOLVER.LR_POLICY.TYPE != 'Plateau':
            val = None

        if lrs_type == cfg.SOLVER.LR_POLICY.SETP_TYPE:
            self.scheduler.step(val)

    def get_lr(self):
        lr = []
        for param_group in self.optimizer.param_groups:
            lr.append(param_group['lr'])
        lr = sorted(list(set(lr)))
        return lr


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)
