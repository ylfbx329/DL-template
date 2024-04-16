import torchvision
from torchinfo import summary

# Total mult-adds (G): 2.37
model = torchvision.models.resnet18(pretrained=True)
summary(model, input_size=(1, 3, 256, 256))

# Total mult-adds (G): 5.34
model = torchvision.models.resnet50(pretrained=True)
summary(model, input_size=(1, 3, 256, 256))

# Total mult-adds (G): 10.19
model = torchvision.models.resnet101(pretrained=True)
summary(model, input_size=(1, 3, 256, 256))

# Total mult-adds (T): 2.32
