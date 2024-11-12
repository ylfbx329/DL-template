import os
import yaml
import wandb
import torch
from torchvision.transforms import v2

from src.config.config import Config


def read_cfg(cfg_path):
    """
    读取YAML格式的配置文件，返回字典
    :param cfg_path: 配置文件完整路径
    :return: 参数字典:
    """
    with open(cfg_path, 'r') as file:
        cfg = yaml.safe_load(file)
    return cfg


def wandb_init():
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


def get_exp_path() -> str:
    """
    获取实验输出目录的绝对路径
    :return:
    """
    # 拼装实验输出目录
    exp_path = os.path.join(Config.args.proj_root, 'outputs', Config.args.exp_name)
    return str(exp_path)


def get_output_path(filename, type):
    """
    获取输出文件的绝对路径
    :param filename: 输出文件的完整文件名
    :param type: 输出文件类型，必须是['checkpoint', 'log', 'result']其中之一
    :return:
    """
    if type not in ['checkpoint', 'log', 'result']:
        raise ValueError(f'type must be "checkpoints" or "logs" or "results", but got {type}')
    path = os.path.join(get_exp_path(), type, filename)
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


def save_ckpt(path, epoch, model, optimizer, loss=None):
    """
    保存检查点
    :param path:
    :param epoch:
    :param model:
    :param optimizer:
    :param loss:
    :return:
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'hyperparameters': {
            'learning_rate': None,
            'batch_size': None,
            'num_epochs': None
        }
    }, path)
