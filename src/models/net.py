import torch.nn as nn
import torchvision
from torchvision.models import ResNet18_Weights

from src.config.config import Config


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        pass
        self.initialize_weights()

    def forward(self, x):
        pass

    def initialize_weights(self):
        pass


def create_model():
    """
    构建模型
    :return:
    """
    net = Net()

    # 测试专用
    net = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
    net.conv1 = nn.Conv2d(Config.args.model.in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                          bias=False)
    net.fc = nn.Linear(512, Config.args.model.num_classes)
    return net
