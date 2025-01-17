import logging

import numpy as np
from tqdm import tqdm

from src.config.config import Config


def train_one_epoch(epoch, model, train_loader, criterion, optimizer, device):
    """
    训练一个epoch
    :param epoch: 第epoch轮训练
    :param model: 模型对象
    :param train_loader: 训练集dataloader
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param device: 训练设备
    :return: 本轮次平均损失
    """
    # 存储每个batch的loss
    loss_history = []

    # 遍历数据集，显示进度条
    total_batch = len(train_loader)  # 便于调用
    log_iter = Config.args.train.log_iter  # 便于调用
    for index, data in tqdm(enumerate(train_loader), desc=f"Epoch {epoch}", total=total_batch):
        # 转移数据
        inputs, labels = [x.to(device) for x in data]

        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)

        # 计算loss
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()

        # 参数更新
        optimizer.step()

        # 记录loss
        loss_history.append(loss.item())

        # 日志打印
        if log_iter != 0 and index % log_iter == 0:
            logging.info(f'Batch [{index}/{total_batch}]: Loss: {np.mean(loss_history)}')

    return np.mean(loss_history)
