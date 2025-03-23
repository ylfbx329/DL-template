import argparse
import os
from pathlib import Path

from src.config.config import Config
from src.exp.base_exp import BaseExp
from src.utils.utils import read_cfg, logging_init, fix_random_seed


def parse_args():
    """
    解析命令行参数，读取配置文件，构建配置类Config
    """
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='Deep learning template project')
    # 必要参数
    # 配置文件
    parser.add_argument('-cfg', default='configs/default.yaml',
                        metavar='<path/to/cfg>', help='path to config file')
    # 可选参数
    parser.add_argument('-train', dest='train_model', action='store_true', default=False,
                        help='train model')
    parser.add_argument('-val', dest='val_model', action='store_true', default=False,
                        help='validate model')
    parser.add_argument('-test', dest='test_model', action='store_true', default=False,
                        help='test model')
    parser.add_argument('-resume', metavar='<ckpt_filename>',
                        help='resume from checkpoint')
    parser.add_argument('-once', action='store_true', default=False,
                        help='test train model structure once')
    # 在此添加更多命令行参数

    # 解析命令行参数
    args = parser.parse_args()

    # 设置工作目录为项目根目录
    args.proj_root = os.getcwd()
    # 设置配置文件名为实验名
    args.exp_name = Path(args.cfg).stem
    # 设置实验输出目录
    args.output_path = Path(args.proj_root, 'outputs', args.exp_name)

    # 读取配置文件
    cfg = read_cfg(args.cfg)

    # 构建配置类Config，使所有参数全局可用
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

    # 创建实验实例
    exp = BaseExp()

    # 模型训练
    if Config.args.train_model:
        exp.train()

    # 模型验证
    if Config.args.val_model:
        exp.validate()

    # 模型测试
    if Config.args.test_model:
        exp.test()


if __name__ == '__main__':
    main()
