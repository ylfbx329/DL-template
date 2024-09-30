from torch.utils.data import DataLoader

from src.data.dataset import MyDataSet


def get_data_loader(data_root, batch_size, split='train', transform=None):
    """
    创建dataloader
    :param data_root:
    :param batch_size:
    :param split:
    :param transform:
    :return:
    """
    # 创建数据集和数据加载器
    dataset = MyDataSet(root=data_root, split=split, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'))
    return dataloader
