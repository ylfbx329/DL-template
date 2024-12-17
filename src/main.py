import argparse
import os

from src.config.config import Config
from src.data.data_loader import get_dataloader
from src.test.test import test
from src.train.train import train
from src.utils.utils import read_cfg, wandb_init, logging_init, fix_random_seed
from src.validate.validate import validate


def parse_args():
    """
    解析命令行、配置文件，构建配置类
    """
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='Deep learning template project')
    # 配置文件
    parser.add_argument('-cfg', default='./configs/default.yaml',
                        help='path to config file')
    # 可选参数
    parser.add_argument('-train', dest='train_model', action='store_true', default=False,
                        help='train model')
    parser.add_argument('-val', dest='val_model', action='store_true', default=False,
                        help='validate model')
    parser.add_argument('-test', dest='test_model', action='store_true', default=False,
                        help='test model')
    parser.add_argument('-wandb', action='store_true', default=False,
                        help='use wandb')
    # 在此添加更多命令行参数

    # 解析参数
    args = parser.parse_args()

    # 设置配置文件名为实验名
    args.exp_name = os.path.basename(args.cfg).rsplit('.', 1)[0]

    # 读取配置文件
    cfg = read_cfg(args.cfg)

    # 设置所有参数全局可用
    Config.update_args([vars(args), cfg])


def main():
    """
    项目主函数
    """
    # 解析参数
    parse_args()

    # 配置logging
    logging_init()

    # 整齐打印参数，必须在配置logging之后
    Config.logging_args()

    # 固定随机种子
    fix_random_seed(seed=Config.args.seed)

    # 启用wandb
    if Config.args.wandb:
        wandb_init()

    # 数据加载
    train_loader, val_loader, test_loader = get_dataloader()

    # 模型训练
    if Config.args.train_model:
        train(train_loader, val_loader)

    # 模型验证
    if Config.args.val_model:
        validate(val_loader)

    # 模型测试
    if Config.args.test_model:
        test(test_loader)


if __name__ == '__main__':
    main()
