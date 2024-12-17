import torch.nn as nn


def get_criterion() -> nn.Module:
    """
    构建loss对象
    :return: 初始化的loss对象
    """
    return nn.CrossEntropyLoss()
