from torch import optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LambdaLR

from src.config.config import Config


def create_optimizer(model,
                     lr=Config.args.optim.lr,
                     optimizer_name=Config.args.optim.name):
    """
    构建优化器和调度器
    :param model:
    :param lr:
    :param optimizer_name:
    :return:
    """
    optim_param = {key: value
                   for key, value in vars(Config.args.optim).items()
                   if key not in ['name', 'lr']}
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, **optim_param)

    if hasattr(Config.args, 'sched'):
        scheduler_name = Config.args.sched.name
        sched_param = {key: value
                       for key, value in vars(Config.args.sched).items()
                       if key != 'name'}
        scheduler = getattr(lr_scheduler, scheduler_name)(optimizer, **sched_param)
    else:
        scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    return optimizer, scheduler
