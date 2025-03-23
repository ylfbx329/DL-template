import torch.nn as nn


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
