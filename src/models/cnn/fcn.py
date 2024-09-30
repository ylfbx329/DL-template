import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict


class FCN(nn.Module):
    """
    自己实现FCN32s

    Args:
        in_channels: 输入通道数
        num_classes: 类别数

    TODO 调整网络结构，验证模型可行性
    """

    def __init__(self, in_channels=3, num_classes=21):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.layer1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=3, padding=100)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)),
            ('relu2', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d(kernel_size=2, stride=2))
        ]))
        self.layer2 = nn.Sequential(OrderedDict([
            ('conv3', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)),
            ('relu3', nn.ReLU(inplace=True)),
            ('conv4', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)),
            ('relu4', nn.ReLU(inplace=True)),
            ('pool2', nn.MaxPool2d(kernel_size=2, stride=2))
        ]))
        self.layer3 = nn.Sequential(OrderedDict([
            ('conv5', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)),
            ('relu5', nn.ReLU(inplace=True)),
            ('conv6', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)),
            ('relu6', nn.ReLU(inplace=True)),
            ('conv7', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)),
            ('relu7', nn.ReLU(inplace=True)),
            ('pool3', nn.MaxPool2d(kernel_size=2, stride=2))
        ]))
        self.layer4 = nn.Sequential(OrderedDict([
            ('conv8', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)),
            ('relu8', nn.ReLU(inplace=True)),
            ('conv9', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)),
            ('relu9', nn.ReLU(inplace=True)),
            ('conv10', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)),
            ('relu10', nn.ReLU(inplace=True)),
            ('pool4', nn.MaxPool2d(kernel_size=2, stride=2))
        ]))
        self.layer5 = nn.Sequential(OrderedDict([
            ('conv11', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)),
            ('relu11', nn.ReLU(inplace=True)),
            ('conv12', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)),
            ('relu12', nn.ReLU(inplace=True)),
            ('conv13', nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)),
            ('relu13', nn.ReLU(inplace=True)),
            ('pool5', nn.MaxPool2d(kernel_size=2, stride=2))
        ]))
        self.fc6 = nn.Sequential(OrderedDict([
            ('conv14', nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=7, padding=0)),
            ('relu14', nn.ReLU(inplace=True)),
            ('dropout1', nn.Dropout(p=0.5, inplace=True))
        ]))
        self.fc7 = nn.Sequential(OrderedDict([
            ('conv15', nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1, padding=0)),
            ('relu15', nn.ReLU(inplace=True)),
            ('dropout2', nn.Dropout(p=0.5, inplace=True))
        ]))
        self.classifier = nn.Conv2d(in_channels=4096, out_channels=self.num_classes, kernel_size=1, padding=0)
        self.init_weights()

    def forward(self, x):
        # 记录输入图像大小
        input_shape = x.shape[-2:]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.classifier(x)
        # 换变量名，否则指向相同空间会报错
        out = x.clone().detach().requires_grad_(True)
        out = F.interpolate(out, size=input_shape, mode="bilinear", align_corners=False)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


if __name__ == '__main__':
    from torchinfo import summary

    model = FCN(num_classes=21)
    summary(model, input_size=(1, 3, 224, 224))
    print(model)
