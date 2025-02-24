from torch import optim, nn
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LambdaLR

from src.config.config import Config


def get_optim_sched(model: nn.Module):
    """
    构建优化器和调度器，根据配置文件自动设置调度器，配置文件中无调度器设置则采用恒定调度器，与不采用调度器效果等同
    :param model: 模型对象
    :return: 优化器和调度器
    """
    optim_param = Config.args.optim
    optim_cls_param = {key: value
                       for key, value in Config.get_argsdict(optim_param).items()
                       if key not in ['name']}
    optim_cls = globals().get(optim_param.name)
    if optim_cls is None:
        optim_cls = getattr(optim, optim_param.name)
    optimizer = optim_cls(model.parameters(), **optim_cls_param)

    if hasattr(Config.args, 'sched'):
        sched_param = Config.args.sched
        sched_cls_param = {key: value
                           for key, value in Config.get_argsdict(sched_param).items()
                           if key not in ['name']}
        sched_cls = globals().get(sched_param.name)
        if sched_cls is None:
            sched_cls = getattr(lr_scheduler, sched_param.name)
        scheduler = sched_cls(optimizer, **sched_cls_param)
    else:
        scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    return optimizer, scheduler
