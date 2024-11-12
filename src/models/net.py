import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet18_Weights

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
    # net = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
    # net.conv1 = nn.Conv2d(Config.args.model.in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # net.fc = nn.Linear(512, Config.args.model.num_classes)
    # print(net)
    if resume:
        ckpt = get_output_path(filename=Config.args.eval.resume, type='checkpoint')
        checkpoint = torch.load(ckpt)
        net.load_state_dict(checkpoint['model_state_dict'])
    return net
