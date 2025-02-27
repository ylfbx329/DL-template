import logging

import numpy as np
import torch
from torchinfo import summary

from src.config.config import Config
from src.criterion.criterion import get_criterion
from src.metrics.metrics import metrics
from src.models.net import create_model
from src.optim.optim import get_optim_sched
from src.train.trainer import train_one_epoch
from src.utils.utils import save_ckpt, load_ckpt
from src.validate.validator import validate_one_epoch
from src.visualize.visualize import plot


def train(train_loader, val_loader):
    """
    训练模型的通用完整流程
    :param train_loader: 训练集dataloader
    :param val_loader: 验证集dataloader，用于对特殊轮次的模型进行验证
    """
    logging.info('Start train...')
    train_param = Config.args.train  # 便于调用

    # 设置训练设备
    device = torch.device(Config.args.device)

    # 初始化模型、损失函数、优化器和调度器
    model = create_model()
    model.to(device)
    criterion = get_criterion()
    optimizer, scheduler = get_optim_sched(model=model)

    # 断点续训
    if Config.args.resume is not None:
        load_ckpt(train_param.ckpt, model, optimizer, scheduler)
    logging.info('model, criterion, optimizer and scheduler create complete.')

    # 设置模型为训练模式
    model.train()

    # 打印模型结构及参数
    # 此处next(iter(train_loader))并不会干扰第一个epoch的迭代，即第一个epoch仍会完整的处理整个数据集
    # 但此处in_data与第一个epoch的第一个batch数据不同，是由于train_loader的shuffle为true，若为false则相同
    in_data, _ = [x.to(device) for x in next(iter(train_loader))]
    summary(model, input_data=in_data)

    # 存储每个epoch的学习率和平均损失
    epoch_lr = []
    epoch_losses = []

    # 记录最好的验证集损失
    best_val_acc = -np.inf

    # 训练模型
    start_epoch = scheduler.last_epoch + 1  # 便于调用
    total_epochs = train_param.epochs  # 便于调用
    for epoch in range(start_epoch, total_epochs):
        # 训练一个epoch，获取epoch平均loss
        epoch_loss = train_one_epoch(epoch, model, train_loader, criterion, optimizer, device)
        epoch_losses.append(epoch_loss)
        epoch_lr.append(scheduler.get_last_lr())

        # 调度器更新，必须在ckpt保存之前，否则调度器的轮次记录与实际训练轮次不符，导致断点续训错误
        scheduler.step()

        # 每个epoch结束后的信息输出，可自定义
        logging.info(f'Epoch [{epoch}/{total_epochs}]: lr: {epoch_lr[-1]}, Loss: {epoch_loss}')

        # 在设定的轮数和训练结束时保存ckpt
        if epoch % train_param.save_epoch == 0 or epoch == total_epochs - 1:
            ckpt_filename = f'epoch{epoch}.pth'
            save_ckpt(ckpt_filename, model, optimizer, scheduler, epoch, epoch_loss)

        # 在设定的轮数和训练结束时使用验证集验证模型
        if epoch >= train_param.val_start and epoch % train_param.val_epoch == 0:
            # 模型验证，返回模型输出、预测结果、预测标签、验证平均损失
            val_output, val_res, val_label, val_loss = validate_one_epoch(model, val_loader, criterion, device)
            # 模型验证信息输出，可自定义
            logging.info(f'Validate: Epoch: {epoch}, Loss: {val_loss}')
            # 计算指标
            val_acc = metrics(val_label, val_res)
            # 保存在验证集指标最优的模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                logging.info(f'Validate: Epoch: {epoch}, Best Accuracy: {val_acc}')
                save_ckpt('best_val.pth', model, optimizer, scheduler, epoch, epoch_loss)

        # 启用wandb时记录日志
        # if Config.args.wandb:
        #     wandb.log({"epoch loss": epoch_loss})

    # epoch-loss图像，可自定义
    if len(epoch_losses) != 0:
        plot(x=range(start_epoch, total_epochs),
             y=epoch_losses,
             xlabel='Epoch',
             ylabel='Loss',
             image_filename=f'epoch{start_epoch}-{total_epochs}_loss.png')

    # epoch-lr图像，可自定义
    if len(epoch_lr) != 0:
        plot(x=range(start_epoch, total_epochs),
             y=epoch_lr,
             xlabel='Epoch',
             ylabel='Learning Rate',
             image_filename=f'epoch{start_epoch}-{total_epochs}_lr.png')

    logging.info('End train!')
