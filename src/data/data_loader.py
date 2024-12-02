from typing import Any

import torchvision
from torch.utils.data import DataLoader

from src.config.config import Config
from src.data.dataset import MyDataSet
from src.utils.utils import get_transform


def get_data_loader(data_root=Config.args.data.root,
                    batch_size=(Config.args.train.batch_size,
                                Config.args.eval.batch_size),
                    num_workers=Config.args.data.num_workers) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """
    创建dataloader
    :param data_root:
    :param batch_size:
    :param num_workers:
    :return:
    """
    # 定义数据变换操作
    transform = get_transform()

    # 创建数据集和数据加载器
    train_set = MyDataSet(root=data_root, split='train', transform=transform)
    test_set = MyDataSet(root=data_root, split='test', transform=transform)

    # 测试专用
    train_set = torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size[0], shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size[1], shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
