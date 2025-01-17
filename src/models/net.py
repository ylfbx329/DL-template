import torch.nn as nn
import torchvision
from torchvision.models import ResNet18_Weights

from src.config.config import Config


class Net(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Net, self).__init__()
        pass
        self.initialize_weights()

    def forward(self, x):
        """
        前向传播
        :param x: 模型输入
        :return: 前向传播结果
        """
        pass

    def initialize_weights(self):
        """
        初始化模型权重
        """
        pass


def create_model() -> nn.Module:
    """
    构建模型
    :return: 模型对象
    """
    model_params = Config.args.model
    net = Net(in_channels=model_params.in_channels,
              num_classes=model_params.num_classes)

    # 测试专用 start
    net = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
    net.conv1 = nn.Conv2d(model_params.in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    net.fc = nn.Linear(512, model_params.num_classes)
    # 测试专用 end

    return net
