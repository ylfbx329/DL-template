import logging
import os

import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2

from src.config.config import Config
from src.data.dataset import MyDataSet


def get_dataloader() -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建dataloader
    :return: 训练集、验证集和测试集的dataloader
    """
    logging.info('Start Data load...')
    data_param = Config.args.data  # 便于调用
    train_param = Config.args.train  # 便于调用
    val_param = Config.args.val  # 便于调用
    test_param = Config.args.test  # 便于调用

    data_root = data_param.root
    batch_size = (train_param.batch_size,
                  val_param.batch_size,
                  test_param.batch_size)
    transform = v2.Compose([v2.PILToTensor(),
                            v2.ConvertImageDtype(torch.float32)])

    # batch_size为int：训练集、验证集和测试集相同
    # batch_size为(int, int)：训练集为batch_size[0]，验证集和测试集为batch_size[1]
    # batch_size为(int, int, int)：训练集、验证集和测试集一一对应
    if isinstance(batch_size, int):
        batch_size = (batch_size, batch_size, batch_size)
    elif isinstance(batch_size, tuple) and len(batch_size) == 2:
        batch_size = (batch_size[0], batch_size[1], batch_size[1])

    # 创建数据集和数据加载器
    train_set = MyDataSet(root=data_root, split='train', transform=transform)
    val_set = MyDataSet(root=data_root, split='val', transform=transform)
    test_set = MyDataSet(root=data_root, split='test', transform=transform)

    # 测试专用 start
    train_set = torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    train_size = int(0.8 * len(train_set))
    val_size = len(train_set) - train_size
    train_set, val_set = random_split(train_set, [train_size, val_size])
    test_set = torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
    # 测试专用 end

    num_workers = min([os.cpu_count(), batch_size[0] if batch_size[0] > 1 else 0, 8]) if os.name == 'posix' else 0
    train_loader = DataLoader(train_set, batch_size=batch_size[0], shuffle=True, num_workers=num_workers)
    num_workers = min([os.cpu_count(), batch_size[1] if batch_size[1] > 1 else 0, 8]) if os.name == 'posix' else 0
    val_loader = DataLoader(val_set, batch_size=batch_size[1], shuffle=False, num_workers=num_workers)
    num_workers = min([os.cpu_count(), batch_size[2] if batch_size[2] > 1 else 0, 8]) if os.name == 'posix' else 0
    test_loader = DataLoader(test_set, batch_size=batch_size[2], shuffle=False, num_workers=num_workers)
    logging.info(f'train sample: {len(train_set)}, val sample: {len(val_set)}, test sample {len(test_set)}')
    logging.info('Data load complete.')
    return train_loader, val_loader, test_loader
