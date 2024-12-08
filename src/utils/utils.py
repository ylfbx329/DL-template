import logging
import os
import random

import numpy as np
import torch
import wandb
import yaml
from torchvision.transforms import v2

from src.config.config import Config


def read_cfg(cfg_path):
    """
    读取YAML格式的配置文件，返回字典
    :param cfg_path: 配置文件完整路径
    :return: 参数字典:
    """
    with open(cfg_path, 'r', encoding='utf-8') as file:
        cfg = yaml.safe_load(file)
    return cfg


def wandb_init():
    """
    初始化WandB
    :return:
    """
    wandb.login()

    project = os.path.basename(Config.args.proj_root)
    argsdict = Config.get_argsdict(Config.args)
    exp_path = get_exp_path()
    wandb.init(
        project=project,
        name=Config.args.exp_name,
        config=argsdict,
        dir=exp_path
    )


def logging_init(filename,
                 level=logging.INFO,
                 mode='a'):
    """
    初始化日志模块
    :param filename:
    :param level:
    :param mode:
    :return:
    """
    logfile = get_output_path(filename, filetype='log')
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logfile, mode=mode),
            logging.StreamHandler()
        ]
    )


def fix_random_seed(seed):
    """
    固定随机种子
    :param seed:
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_exp_path() -> str:
    """
    获取实验输出目录的绝对路径
    :return:
    """
    # 拼装实验输出目录
    exp_path = os.path.join(Config.args.proj_root, 'outputs', Config.args.exp_name)
    return str(exp_path)


def get_output_path(filename, filetype):
    """
    获取输出文件的绝对路径
    :param filename: 输出文件的完整文件名
    :param filetype: 输出文件类型，必须是['checkpoint', 'log', 'result']其中之一
    :return:
    """
    assert filetype in ['checkpoint', 'log',
                        'result'], f'type must be "checkpoint" or "log" or "result", but got {filetype}'
    path = os.path.join(get_exp_path(), filetype, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def get_transform():
    """
    定义变换操作，可自定义修改

    尽量使用torchvision.transforms.v2接口定义操作
    :return: 变换操作
    """
    transform = v2.Compose([
        v2.PILToTensor(),
        v2.ConvertImageDtype(torch.float32),
    ])
    return transform


def save_ckpt(ckpt_filename, epoch, model, optimizer, scheduler, loss=None):
    """
    保存ckpt
    :param ckpt_filename:
    :param epoch:
    :param model:
    :param optimizer:
    :param scheduler:
    :param loss:
    :return:
    """
    path = get_output_path(ckpt_filename, filetype='checkpoint')
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'loss': loss
    }, path)
    logging.info(f'Save checkpoint at {path}')


def load_ckpt(ckpt_filename, model, optimizer=None, scheduler=None):
    """
    加载ckpt
    :param ckpt_filename:
    :param model:
    :param optimizer:
    :param scheduler:
    :return:
    """
    path = get_output_path(ckpt_filename, filetype='checkpoint')
    assert os.path.exists(path), f'checkpoint: {path} not exist!'
    # 指定map_location='cpu'避免显存占用，显示指定weights_only以兼容后续pytorch版本
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    logging.info(f'Load checkpoint: {path}, epoch: {checkpoint["epoch"]}, train loss: {checkpoint["loss"]}')
