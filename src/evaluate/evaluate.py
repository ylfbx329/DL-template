import numpy as np
import torch
import wandb
from tqdm import tqdm

from src.config.config import Config
from src.data.data_loader import get_data_loader
from src.models.net import create_model
from src.criterion.criterion import create_criterion
from src.utils.utils import get_transform, get_output_path


@torch.no_grad()
def evaluate():
    """
    评估模型性能完整流程
    :return:
    """
    print('Start evaluate...')
    # 参数设置
    data_param = Config.args.data
    eval_param = Config.args.eval

    # 评估设备设置
    device = torch.device(Config.args.device)

    # 定义数据变换操作
    transform = get_transform()

    # 数据加载对象
    test_loader = get_data_loader(data_param.root, eval_param.batch_size, split='test', transform=transform)

    # 加载模型、初始化损失函数
    model = create_model(resume=True)
    criterion = create_criterion()

    # 设置评估模式
    model.eval()

    # 转移模型
    model.to(device)

    # 存储每个batch的loss
    loss_history = []

    # 正确case计数
    acc_sum = 0

    # 结果保存
    results = []

    if Config.args.use_wandb:
        table = wandb.Table(columns=["accuracy", "loss"])

    # 测试模型，遍历数据集，显示进度条
    total_batch = len(test_loader)  # 便于调用
    for index, data in tqdm(enumerate(test_loader), desc=f"Evaluate", total=total_batch):
        # 转移数据
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # 前向传播
        outputs = model(inputs)

        # 计算loss
        loss = criterion(outputs, labels)

        # 获取分类结果
        result = torch.argmax(outputs, dim=1)

        # 正确case计数
        acc = (result == labels).sum().item()

        acc_sum += acc

        # 记录loss
        loss_history.append(loss.item())

        # 结果保存
        results.append(result.cpu().numpy())

        if Config.args.use_wandb:
            table.add_data(acc, loss)

    if Config.args.use_wandb:
        wandb.log({"table_key": table})

    # 计算正确率
    acc_rate = acc_sum / len(test_loader.dataset)

    # 计算平均loss
    eval_loss = np.mean(loss_history)

    # 保存结果
    results = np.concatenate(results)
    resume = eval_param.resume.split(".")[0]
    np.save(get_output_path(filename=f'{resume}-results.npy', type='result'), results)

    # 信息输出，可自定义
    print(f'Evaluate: Accuracy={acc_rate}: Loss: {eval_loss}')

    print('End evaluate!')
