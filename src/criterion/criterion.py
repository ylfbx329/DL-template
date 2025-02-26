import logging

import torch.nn as nn

from src.config.config import Config


def get_criterion() -> nn.Module:
    """
    构建loss对象
    :return: 初始化的loss对象
    """
    loss_param = Config.args.loss
    # 获取loss所需参数
    loss_cls_param = {key: value
                      for key, value in Config.get_argsdict(loss_param).items()
                      if key not in ['name']}
    # 根据损失函数名查找类名，得到类对象
    # 在项目内查找
    loss_cls = globals().get(loss_param.name)
    # 若未找到，则在pytorch的包内寻找（torch.nn）
    if loss_cls is None:
        try:
            loss_cls = getattr(nn, loss_param.name)
        except AttributeError:
            raise AttributeError(f'Loss class "{loss_param.name}" not found. Please ensure the class is imported correctly in {__file__}.')
    # 传递参数，实例化类对象
    criterion = loss_cls(**loss_cls_param)

    # 打印类信息，防止错误构建
    logging.info(f'create criterion: {criterion.__class__}')
    return criterion
