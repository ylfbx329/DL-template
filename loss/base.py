from torch import nn


class BaseLoss(nn.Module):
    def __init__(self):
        super(BaseLoss, self).__init__()
        self.identity = nn.Identity()

    def forward(self, output, target):
        return self.identity(target - output)
