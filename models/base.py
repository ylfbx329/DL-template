from torch import nn


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.identity = nn.Identity()

    def forward(self, x):
        return self.identity(x)
