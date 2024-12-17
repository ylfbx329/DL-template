from torch.utils.data import Dataset


class MyDataSet(Dataset):
    def __init__(self, root, split, transform=None):
        super(Dataset, self).__init__()
        pass

    def __getitem__(self, index):
        """
        根据索引返回<样本，标签>对，可同时对数据进行变换
        :param index: 索引
        :return: 样本和标签
        """
        pass

    def __len__(self):
        """
        返回数据集长度，即有多少个<样本，标签>对
        :return: 数据集长度
        """
        pass
