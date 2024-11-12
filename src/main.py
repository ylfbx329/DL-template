import argparse

from src.config.config import Config
from src.evaluate.evaluate import evaluate
from src.train.train import train
from src.utils.utils import read_cfg, wandb_init


def parse_args():
    """
    解析命令行参数
    """
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='Deep learning template project')
    # 必要参数
    parser.add_argument('cfg', default='./configs/default.yaml',
                        help='path to config file')
    # 可选参数
    parser.add_argument('--is_train', action='store_true', default=False,
                        help='train model')
    parser.add_argument('--is_eval', action='store_true', default=False,
                        help='eval model')
    # 在此添加更多命令行参数

    # 解析参数
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # 解析命令行参数
    args = parse_args()
    # 读取配置文件
    cfg = read_cfg(args.cfg)

    # 设置所有参数全局可用
    Config.update_args([vars(args), cfg])

    # 整齐打印参数
    Config.print_args()

    if Config.args.use_wandb:
        wandb_init()

    # 模型训练
    if args.is_train:
        train()

    # 模型验证
    if args.is_eval:
        evaluate()
