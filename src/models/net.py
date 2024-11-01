import torch
import torch.nn as nn

from src.config.config import Config
from src.utils.utils import get_output_path


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        pass
        self.initialize_weights()

    def forward(self, x):
        pass
        return x

    def initialize_weights(self):
        pass


def create_model(resume=False):
    """
    创建模型，选择性加载模型权重
    :return:
    """
    net = Net()
    if resume:
        ckpt = get_output_path(filename=Config.args.eval.resume, type='checkpoint')
        checkpoint = torch.load(ckpt)
        net.load_state_dict(checkpoint['model_state_dict'])
    return net
