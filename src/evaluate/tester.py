import numpy as np
import torch
from tqdm import tqdm


@torch.no_grad()
def test_once(model, test_loader, criterion, device):
    """
    用完整数据验证模型
    :param model:
    :param test_loader:
    :param criterion:
    :param device:
    :return: 平均损失、正确率、预测结果
    """
    # 存储每个batch的loss
    loss_history = []

    # 正确case计数
    acc = 0

    # 结果保存
    results = []

    # 遍历数据集，显示进度条
    total_batch = len(test_loader)  # 便于调用
    for index, data in tqdm(enumerate(test_loader), desc=f"Evaluate", total=total_batch):
        # 转移数据
        inputs, labels = [x.to(device) for x in data]

        # 前向传播
        outputs = model(inputs)

        # 计算loss
        loss = criterion(outputs, labels)

        # 解码预测结果
        result = torch.argmax(outputs, dim=1)

        # 正确case计数
        acc += (result == labels).sum().item()

        # 记录loss
        loss_history.append(loss.item())

        # 结果保存
        results.append(result.cpu().numpy())

    # 计算正确率
    acc_rate = acc / len(test_loader.dataset)

    # 计算平均loss
    eval_loss = np.mean(loss_history)

    return eval_loss, acc_rate, results
