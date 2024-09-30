import os
import yaml
import torch
from torchvision.transforms import v2


def read_cfg(cfg_path):
    """
    读取配置文件，返回字典
    :param cfg_path:
    :return: 参数字典:
    """
    with open(cfg_path, 'r') as file:
        cfg = yaml.safe_load(file)
    return cfg


def get_exp_path(exp_name):
    """
    获取实验输出目录
    :param exp_name:
    :return:
    """
    # 拼装实验输出目录
    exp_path = os.path.join(os.environ['PROJECT_ROOT'], 'outputs', exp_name)

    return exp_path


def get_ckpt_path(exp_name, ckpt_name):
    """
    获取检查点保存路径
    :param exp_name:本次实验名称，用以区分各次实验
    :param ckpt_name:检查点文件名
    :return: 完整的检查点保存路径
    """
    # 拼装checkpoint保存目录
    ckpt_dir = os.path.join(get_exp_path(exp_name), 'checkpoints')

    # 确保目录存在，如果不存在则创建
    os.makedirs(ckpt_dir, exist_ok=True)

    # 拼装checkpoint路径
    ckpt_path = os.path.join(ckpt_dir, f'{ckpt_name}.pth')
    return ckpt_path


def get_res_path(exp_name, result_file_name):
    """
    获取结果文件保存路径
    :param exp_name:本次实验名称，用以区分各次实验
    :param result_file_name:结果文件名
    :return: 完整的结果文件保存路径
    """
    # 拼装结果文件保存目录
    res_dir = os.path.join(get_exp_path(exp_name), 'results')

    # 确保目录存在，如果不存在则创建
    os.makedirs(res_dir, exist_ok=True)

    # 拼装结果文件路径
    res_path = os.path.join(res_dir, result_file_name)
    return res_path


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
