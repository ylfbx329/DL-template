from torch import optim

from src.config.config import Config


def create_optimizer(model,
                     lr=Config.args.train.lr):
    """
    返回优化器对象
    :param model:
    :param lr:
    :return:
    """
    return optim.Adam(model.parameters(), lr=lr)
