import logging
import os

import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2

from src.config.config import Config
from src.data.dataset import MyDataSet
from src.data.utils import seed_worker


def get_dataloader():
    """
    组织数据，创建dataloader
    :return: 训练集、验证集和测试集的dataloader
    """
    logging.info('Start Data load...')
    data_param = Config.args.data
    train_param = Config.args.train
    val_param = Config.args.val
    test_param = Config.args.test

    # 数据变换，按需修改
    transform = v2.Compose([v2.PILToTensor(),
                            v2.ConvertImageDtype(torch.float32)])

    # 创建数据集和数据加载器
    train_set = MyDataSet(root=data_param.root, split='train', transform=transform)
    val_set = MyDataSet(root=data_param.root, split='val', transform=transform)
    test_set = MyDataSet(root=data_param.root, split='test', transform=transform)

    # 测试专用 start
    train_set = torchvision.datasets.MNIST(root=data_param.root, train=True, download=True, transform=transform)
    train_size = int(0.8 * len(train_set))
    val_size = len(train_set) - train_size
    train_set, val_set = random_split(train_set, [train_size, val_size])
    test_set = torchvision.datasets.MNIST(root=data_param.root, train=False, download=True, transform=transform)
    # 测试专用 end

    # num_workers的通用设置方法，Windows为0，Linux为CPU逻辑核心数和batch_size的小值，最大不超过8
    num_workers = min([os.cpu_count(), train_param.batch_size if train_param.batch_size > 1 else 0, 8]) if os.name == 'posix' else 0
    # 当使用GPU训练时pin_memory设置为True，提高数据加载效率
    pin_memory = True if 'cuda' in Config.args.device else False
    # 避免数据加载的随机性
    g = torch.Generator()
    g.manual_seed(0)

    # 创建dataloader
    # 设置worker_init_fn以保证实验的可重复性
    train_loader = DataLoader(train_set,
                              batch_size=train_param.batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              worker_init_fn=seed_worker,
                              generator=g)
    val_loader = DataLoader(val_set,
                            batch_size=val_param.batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            worker_init_fn=seed_worker,
                            generator=g)
    test_loader = DataLoader(test_set,
                             batch_size=test_param.batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=pin_memory,
                             worker_init_fn=seed_worker,
                             generator=g)

    # 日志打印，可自定义
    logging.info(f'train sample: {len(train_set)}, val sample: {len(val_set)}, test sample {len(test_set)}')
    logging.info('Data load complete.')
    return train_loader, val_loader, test_loader
