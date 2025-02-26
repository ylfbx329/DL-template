import logging

import torch

from src.config.config import Config
from src.criterion.criterion import get_criterion
from src.metrics.metrics import metrics
from src.models.net import create_model
from src.utils.utils import load_ckpt
from src.validate.validator import validate_one_epoch


@torch.no_grad()
def validate(val_loader):
    """
    验证模型的通用完整流程
    :param val_loader: 验证集dataloader
    """
    logging.info('Start validate...')
    val_param = Config.args.val  # 便于调用

    # 设置验证设备
    device = torch.device(Config.args.device)

    # 初始化模型、损失函数
    model = create_model()
    model.to(device)
    criterion = get_criterion()

    # 加载ckpt
    load_ckpt(val_param.ckpt, model)
    logging.info('model and criterion create complete.')

    # 设置模型为评估模式
    model.eval()

    # 使用验证集验证模型性能，得到模型输出，预测结果，预测标签，平均损失
    output, result, label, loss = validate_one_epoch(model, val_loader, criterion, device)

    # 信息输出，可自定义
    logging.info(f'Validate: Loss: {loss}')

    # 计算指标
    metrics(label, result)

    logging.info('End validate!')
