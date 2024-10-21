from torch import optim


def create_optimizer(model, learning_rate):
    """
    返回优化器对象
    :return:
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return optimizer
