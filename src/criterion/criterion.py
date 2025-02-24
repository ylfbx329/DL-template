import torch.nn as nn

from src.config.config import Config


def get_criterion() -> nn.Module:
    """
    构建loss对象
    :return: 初始化的loss对象
    """
    loss_param = Config.args.loss
    loss_cls_param = {key: value
                      for key, value in Config.get_argsdict(loss_param).items()
                      if key not in ['name']}
    loss_cls = globals().get(loss_param.name)
    if loss_cls is None:
        loss_cls = getattr(nn, loss_param.name)
    return loss_cls(**loss_cls_param)
