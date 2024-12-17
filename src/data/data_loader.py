import logging

import torch
from torch.utils.data import DataLoader
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
    num_workers = data_param.num_workers
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

    train_loader = DataLoader(train_set, batch_size=batch_size[0], shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size[1], shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size[2], shuffle=False, num_workers=num_workers)
    logging.info(f'train sample: {len(train_set)}, val sample: {len(val_set)}, test sample {len(test_set)}')
    logging.info('Data load complete.')
    return train_loader, val_loader, test_loader
