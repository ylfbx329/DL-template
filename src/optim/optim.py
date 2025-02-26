import logging

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
    # 获取optimizer所需参数
    optim_cls_param = {key: value
                       for key, value in Config.get_argsdict(optim_param).items()
                       if key not in ['name']}
    # 根据优化器名查找类名，得到类对象
    optim_cls = globals().get(optim_param.name)
    # 若未找到，则在pytorch的包内寻找（torch.optim）
    if optim_cls is None:
        try:
            optim_cls = getattr(optim, optim_param.name)
        except AttributeError:
            raise AttributeError(f'Optimizer class "{optim_param.name}" not found. Please ensure the class is imported correctly in {__file__}.')
    # 传递参数，实例化类对象
    optimizer = optim_cls(model.parameters(), **optim_cls_param)

    if hasattr(Config.args, 'sched'):
        sched_param = Config.args.sched
        # 获取optimizer所需参数
        sched_cls_param = {key: value
                           for key, value in Config.get_argsdict(sched_param).items()
                           if key not in ['name']}
        # 根据优化器名查找类名，得到类对象
        sched_cls = globals().get(sched_param.name)
        # 若未找到，则在pytorch的包内寻找（torch.optim.lr_scheduler）
        if sched_cls is None:
            try:
                sched_cls = getattr(lr_scheduler, sched_param.name)
            except AttributeError:
                raise AttributeError(f'Scheduler class "{sched_param.name}" not found. Please ensure the class is imported correctly in {__file__}.')
        # 传递参数，实例化类对象
        scheduler = sched_cls(optimizer, **sched_cls_param)
    else:
        # 默认调度器不会改变学习率，用于方便设计统一的训练流程
        scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    # 打印类信息，防止错误构建
    logging.info(f'create optimizer and scheduler: {optimizer.__class__}, {scheduler.__class__}')
    return optimizer, scheduler
