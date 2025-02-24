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
    model_param = Config.args.model
    net = Net(in_channels=model_param.in_channels,
              num_classes=model_param.num_classes)

    model_cls_param = {key: value
                       for key, value in Config.get_argsdict(model_param).items()
                       if key not in ['name']}
    model_cls = globals().get(model_param.name)
    model = model_cls(**model_cls_param)

    # 测试专用 start
    model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.conv1 = nn.Conv2d(model_param.in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(512, model_param.num_classes)
    # 测试专用 end

    return model
