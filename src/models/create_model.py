import torch

from src.models.net import Net


def create_model(param, resume=None):
    """
    返回初始化模型
    :return:
    """
    net = Net(param)
    if resume is not None:
        checkpoint = torch.load(resume)
        net.load_state_dict(checkpoint['model_state_dict'])
    return net
