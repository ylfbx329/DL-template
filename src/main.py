import argparse
import os

from src.config.config import Config
from src.utils.utils import read_cfg, wandb_init, logging_init, fix_random_seed


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
    parser.add_argument('-test', dest='eval_model', action='store_true', default=False,
                        help='eval model')
    parser.add_argument('-wandb', action='store_true', default=False,
                        help='use wandb')
    # 在此添加更多命令行参数

    # 解析参数
    args = parser.parse_args()

    # 设置配置文件名为实验名
    args.exp_name = os.path.basename(args.cfg).split('.')[0]

    # 读取配置文件
    cfg = read_cfg(args.cfg)

    # 设置所有参数全局可用
    Config.update_args([vars(args), cfg])


def main():
    """
    项目主函数
    """
    # 解析命令行参数
    parse_args()

    # 配置logging
    mode = []
    if Config.args.train_model:
        mode.append('_train')
    if Config.args.eval_model:
        mode.append('_eval')
    logfile = Config.args.exp_name + ''.join(mode) + '.log'
    logging_init(logfile)

    # 整齐打印参数
    Config.logging_args()

    # 固定随机种子
    fix_random_seed(Config.args.seed)

    # Config类创建后引入，避免函数声明失败
    from src.train.train import train
    from src.evaluate.evaluate import evaluate

    # 启用wandb
    if Config.args.wandb:
        wandb_init()

    # 模型训练
    if Config.args.train_model:
        train()

    # 模型验证
    if Config.args.eval_model:
        evaluate()


if __name__ == '__main__':
    main()
