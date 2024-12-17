from torch import optim, nn
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LambdaLR

from src.config.config import Config


def get_optim_sched(optimizer_name: str,
                    model: nn.Module,
                    lr: float):
    """
    构建优化器和调度器，根据配置文件自动设置调度器，配置文件中无调度器设置则采用恒定调度器，与不采用调度器效果等同
    :param optimizer_name: 优化器类名
    :param model: 模型对象
    :param lr: 学习率
    :return: 优化器和调度器
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
