import pprint
import argparse
import matplotlib.pyplot as plt
import torch

from src.data.data_loader import get_data_loader
from src.train.trainer import train_one_epoch
from src.models.create_model import create_model
from src.criterion.create_loss import create_criterion
from src.optim.create_optimizer import create_optimizer
from src.utils.utils import read_cfg, get_ckpt_path, save_ckpt, get_transform, get_res_path


def train(param):
    """
    训练模型的完整流程
    :param param:
    :return:
    """
    print('Start train...')
    # 参数设置
    data_param = param['data']
    model_param = param['model']
    train_param = param['train']

    # 训练设备设置
    device = torch.device(train_param['device'])

    # 定义数据变换操作
    transform = get_transform()

    # 数据加载对象
    train_loader = get_data_loader(data_param['root'], train_param['batch_size'], split='train', transform=transform)

    # 初始化模型、损失函数和优化器
    model = create_model(model_param)
    criterion = create_criterion()
    optimizer = create_optimizer(model, train_param['learning_rate'])

    # 设置训练模式
    model.train()

    # 转移模型
    model.to(device)

    # 存储每个epoch的平均loss
    epoch_losses = []

    # 训练模型
    total_epochs = train_param['epochs']  # 便于调用
    for epoch in range(total_epochs):
        # 训练一个epoch，获取epoch平均loss
        epoch_loss = train_one_epoch(epoch, model, train_loader, criterion, optimizer, device)
        epoch_losses.append(epoch_loss)

        # 保存此轮模型
        ckpt_path = get_ckpt_path(param['exp_name'], ckpt_name=f'epoch{epoch}')
        save_ckpt(ckpt_path, epoch, model, optimizer, epoch_loss)

        # 信息输出，可自定义
        print(f'Epoch [{epoch}/{total_epochs}]: Loss: {epoch_loss}')
        print(f'Save model at {ckpt_path}')

    # epoch-loss图像，可自定义
    plt.plot(epoch_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xlim([-2, total_epochs + 2])
    plt.savefig(get_res_path(param['exp_name'], 'loss.png'))
    plt.show()
    print('End train!')


if __name__ == '__main__':
    # 设置命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', help='path to config file')  # 必要参数
    # 在此添加更多命令行参数

    # 转为参数字典
    args = parser.parse_args()

    # 读取配置文件
    param = read_cfg(args.cfg)

    # 更新参数字典
    param.update(vars(args))

    # 整齐打印参数
    pprint.pprint(param)

    # 模型训练
    train(param)
