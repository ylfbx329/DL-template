import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from src.config.config import Config


def read_cfg(cfg_path):
    """
    读取YAML格式的配置文件，返回字典
    :param cfg_path: 配置文件路径
    :return: 参数字典
    """
    with open(cfg_path, 'r', encoding='utf-8') as file:
        cfg = yaml.safe_load(file)
    return cfg


def logging_init(log_filename=None,
                 level=logging.INFO,
                 mode='a'):
    """
    初始化logging模块。
    设置日志文件路径、日志级别，日志消息样式、输出流（控制台与日志文件同时输出）
    :param log_filename: 日志文件名，默认为”配置文件名_运行模式.log“
    :param level: 日志级别，默认为INFO
    :param mode: 写入模式，默认为追加
    """
    if log_filename is None:
        suffix = ''
        suffix += '_train' if Config.args.train_model else ''
        suffix += '_val' if Config.args.val_model else ''
        suffix += '_test' if Config.args.test_model else ''
        log_filename = Config.args.exp_name + suffix + '.log'
    log_filepath = get_output_path(log_filename, filetype='log')
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath, mode=mode),
            logging.StreamHandler()
        ]
    )


def fix_random_seed(seed):
    """
    固定随机种子
    :param seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  # 禁用cuDNN基准测试
    torch.backends.cudnn.deterministic = True  # 强制使用确定性卷积算法


def get_output_path(filename, filetype):
    """
    获取输出文件的绝对路径，创建父目录
    :param filename: 输出文件的完整文件名
    :param filetype: 输出文件类型，必须是['checkpoint', 'log', 'result']其中之一
    :return: 输出文件的绝对路径
    """
    assert filetype in ['checkpoint', 'log', 'result'], f'filetype must be in ["checkpoint", "log", "result"], but got {filetype}'
    path = Path(Config.args.output_path, filetype, filename)
    os.makedirs(path.parent, exist_ok=True)
    return path


def save_ckpt(ckpt_filename: str,
              model: nn.Module,
              optimizer: Optimizer = None,
              scheduler: LRScheduler = None,
              epoch: int = None,
              loss: float = None):
    """
    训练过程中保存ckpt
    :param ckpt_filename: ckpt文件名
    :param epoch: ckpt所属的训练轮次
    :param model: 模型对象
    :param optimizer: 优化器对象
    :param scheduler: 调度器对象
    :param loss: 本轮训练损失
    """
    # 获取ckpt保存路径
    path = get_output_path(ckpt_filename, filetype='checkpoint')
    # 保存ckpt
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'loss': loss
    }, path)
    # 打印保存信息
    logging.info(f'Save checkpoint at {path}')


def load_ckpt(ckpt_filename: str,
              model: nn.Module,
              optimizer: Optimizer = None,
              scheduler: LRScheduler = None):
    """
    为模型、优化器和调度器加载ckpt
    :param ckpt_filename: ckpt文件名
    :param model: 模型对象
    :param optimizer: 优化器对象，仅用于训练阶段
    :param scheduler: 调度器对象，仅用于训练阶段
    """
    path = get_output_path(ckpt_filename, filetype='checkpoint')
    assert path.exists(), f'checkpoint: {path} not exist!'
    # 指定map_location='cpu'避免显存占用，显示指定weights_only以兼容后续pytorch版本
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    logging.info(f'Load checkpoint: {path}, epoch: {checkpoint["epoch"]}, train loss: {checkpoint["loss"]}')
