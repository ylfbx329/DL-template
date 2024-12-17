import logging

import numpy as np
import torch
from tqdm import tqdm

from src.config.config import Config


@torch.no_grad()
def validate_one_epoch(model, val_loader, criterion, device) -> tuple[np.ndarray, np.ndarray, float]:
    """
    使用验证集评估模型性能
    :param model: 模型对象
    :param val_loader: 验证集dataloader
    :param criterion: 损失函数
    :param device: 验证设备
    :return: 预测结果，标签，平均损失
    """
    # 存储每个batch的loss
    loss_history = []

    # 保存结果和标签
    res_list = []
    label_list = []

    # 遍历数据集，显示进度条
    total_batch = len(val_loader)  # 便于调用
    log_iter = Config.args.val.log_iter  # 便于调用
    for index, data in tqdm(enumerate(val_loader), desc="Validate", total=total_batch):
        # 转移数据
        inputs, labels = [x.to(device) for x in data]

        # 前向传播
        outputs = model(inputs)

        # 计算loss
        loss = criterion(outputs, labels)

        # 模型输出后处理，可自定义
        result = torch.argmax(outputs, dim=1)

        # 记录结果
        res_list.append(result.cpu().numpy())

        # 记录标签
        label_list.append(labels.cpu().numpy())

        # 记录loss
        loss_history.append(loss.item())

        # 日志打印
        if log_iter != 0 and index % log_iter == 0:
            logging.info(f'Batch [{index}/{total_batch}]: Loss: {np.mean(loss_history)}')

    res = np.concatenate(res_list)
    label = np.concatenate(label_list)
    avg_loss = np.mean(loss_history)
    return res, label, avg_loss
