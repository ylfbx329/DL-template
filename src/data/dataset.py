from torch.utils.data import Dataset

from src.config.config import Config


class MyDataSet(Dataset):
    def __init__(self,
                 root=Config.args.data.root,
                 split='train',
                 transform=None):
        super(Dataset, self).__init__()
        pass

    def __getitem__(self, index):
        """
        [Optional]对样本和标签进行数据变换
        返回<样本，标签>对
        :param index:
        :return:
        """
        pass

    def __len__(self):
        """
        返回数据集长度，即有多少个<样本，标签>对
        :return:
        """
        pass
