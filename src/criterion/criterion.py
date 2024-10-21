import torch.nn as nn


def create_criterion():
    """
    返回初始化loss对象
    :return:
    """
    return nn.CrossEntropyLoss()
