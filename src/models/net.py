import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, param):
        super(Net, self).__init__()
        pass
        self.initialize_weights()

    def forward(self, x):
        pass
        return x

    def initialize_weights(self):
        pass

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