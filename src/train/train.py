import logging

import matplotlib.pyplot as plt
import torch
import wandb
from torchinfo import summary

from src.config.config import Config
from src.criterion.criterion import create_criterion
from src.data.data_loader import get_data_loader
from src.models.net import create_model
from src.optim.optim import create_optimizer
from src.train.trainer import train_one_epoch
from src.utils.utils import save_ckpt, get_output_path, load_ckpt


def train():
    """
    训练模型的完整流程
    """
    logging.info('Start train...')
    train_param = Config.args.train  # 便于调用

    # 训练设备设置
    device = torch.device(Config.args.device)

    # 数据加载对象
    train_loader, test_loader = get_data_loader()
    logging.info('Data load complete.')

    # 初始化模型、损失函数和优化器
    model = create_model()
    model.to(device)
    criterion = create_criterion()
    optimizer, scheduler = create_optimizer(model)
    if hasattr(train_param, 'ckpt'):
        load_ckpt(train_param.ckpt, model, optimizer, scheduler)
    logging.info('model, criterion, optimizer and scheduler create complete.')

    # 设置训练模式
    model.train()

    # 打印模型参数
    in_data, _ = [x.to(device) for x in next(iter(train_loader))]
    summary(model, input_data=in_data)

    # 存储每个epoch的平均loss和lr
    epoch_losses = []
    epoch_lr = []

    # 训练模型
    total_epochs = train_param.epochs  # 便于调用
    start_epoch = scheduler.last_epoch + 1  # 便于调用
    for epoch in range(start_epoch, total_epochs):
        # 训练一个epoch，获取epoch平均loss
        epoch_loss = train_one_epoch(epoch, model, train_loader, criterion, optimizer, device)
        epoch_losses.append(epoch_loss)
        epoch_lr.append(scheduler.get_last_lr())

        # 调度器更新，必须在ckpt保存之前
        scheduler.step()

        # 每个epoch结束后的信息输出，可自定义
        logging.info(f'Epoch [{epoch}/{total_epochs}]: lr: {epoch_lr[-1]}, Loss: {epoch_loss}')

        # 在设定的轮数和训练结束时保存ckpt
        if epoch % train_param.save_freq == 0 or epoch == total_epochs - 1:
            # 保存此轮模型
            ckpt_filename = f'epoch{epoch}.pth'
            save_ckpt(ckpt_filename, epoch, model, optimizer, scheduler, epoch_loss)

        if Config.args.wandb:
            wandb.log({"epoch loss": epoch_loss})

    # epoch-loss图像，可自定义
    if len(epoch_losses) != 0:
        plt.plot(epoch_losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.xlim([start_epoch - 2, total_epochs + 2])
        filename = f'epoch{start_epoch}-{total_epochs}_loss.png'
        plt.savefig(get_output_path(filename=filename, filetype='result'))
        plt.show()

    # epoch-lr图像，可自定义
    if len(epoch_lr) != 0:
        plt.plot(epoch_lr)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.xlim([start_epoch - 2, total_epochs + 2])
        filename = f'epoch{start_epoch}-{total_epochs}_lr.png'
        plt.savefig(get_output_path(filename=filename, filetype='result'))
        plt.show()

    logging.info('End train!')
