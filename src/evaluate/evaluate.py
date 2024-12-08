import logging

import numpy as np
import torch

from src.config.config import Config
from src.criterion.criterion import create_criterion
from src.data.data_loader import get_data_loader
from src.evaluate.tester import test_once
from src.models.net import create_model
from src.utils.utils import get_output_path, load_ckpt


@torch.no_grad()
def evaluate():
    """
    评估模型性能完整流程
    """
    logging.info('Start evaluate...')
    eval_param = Config.args.eval  # 便于调用

    # 评估设备设置
    device = torch.device(Config.args.device)

    # 数据加载对象
    train_loader, test_loader = get_data_loader()
    logging.info('Data load complete.')

    # 加载模型、初始化损失函数
    model = create_model()
    model.to(device)
    criterion = create_criterion()
    load_ckpt(eval_param.ckpt, model)
    logging.info('model and criterion create complete.')

    # 设置评估模式
    model.eval()

    # 用完整数据验证模型，得到平均loss、正确率、模型输出结果
    eval_loss, acc_rate, results = test_once(model, test_loader, criterion, device)
    # 信息输出，可自定义
    logging.info(f'Evaluate: Accuracy={acc_rate}: Loss: {eval_loss}')

    # 保存结果
    results = np.concatenate(results)
    ckpt = eval_param.ckpt.split(".")[0]
    res_path = get_output_path(filename=f'{ckpt}-results.npy', filetype='result')
    np.save(res_path, results)
    logging.info(f'Save results at {res_path}')

    logging.info('End evaluate!')
