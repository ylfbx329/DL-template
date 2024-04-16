import torch.utils.data as data

class MNIST(data.Dataset):
    def __init__(self):
        super(MNIST, self).__init__()
        return torchvision.datasets.MNIST