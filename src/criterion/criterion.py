import torch.nn as nn


class CustomLoss(nn.Module):
    def __init__(self):
        """
        自定义损失函数
        """
        super(CustomLoss, self).__init__()

    def forward(self, outputs, targets):
        """
        计算损失
        :param outputs: 模型的输出
        :param targets: 真实标签
        :return: 计算的损失值
        """
        pass
