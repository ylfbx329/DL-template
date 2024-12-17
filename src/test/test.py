import logging

import numpy as np
import torch

from src.config.config import Config
from src.criterion.criterion import get_criterion
from src.metrics.metrics import metrics
from src.models.net import create_model
from src.test.tester import test_one_epoch
from src.utils.utils import get_output_path, load_ckpt


@torch.no_grad()
def test(test_loader):
    """
    测试模型的完整流程
    :param test_loader: 测试集dataloader
    """
    logging.info('Start test...')
    test_param = Config.args.test  # 便于调用

    # 评估设备设置
    device = torch.device(Config.args.device)

    # 初始化模型、损失函数
    model = create_model()
    model.to(device)
    criterion = get_criterion()

    # 加载ckpt
    load_ckpt(test_param.ckpt, model)
    logging.info('model and criterion create complete.')

    # 设置模型为评估模式
    model.eval()

    # 使用测试集测试模型性能，得到预测结果，标签，平均损失
    result, label, loss = test_one_epoch(model, test_loader, criterion, device)

    # 信息输出，可自定义
    logging.info(f'Test: Loss: {loss}')

    # 计算指标
    metrics(label, result)

    # 保存结果
    ckpt = test_param.ckpt.split('.')[0]
    res_path = get_output_path(filename=f'{ckpt}-result.npy', filetype='result')
    label_path = get_output_path(filename=f'{ckpt}-label.npy', filetype='result')
    np.save(res_path, result)
    np.save(label_path, label)
    logging.info(f'Save result at {res_path} and label at {label_path}')

    logging.info('End test!')
