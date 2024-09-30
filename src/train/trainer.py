from tqdm import tqdm


def train_one_epoch(epoch, model, train_loader, criterion, optimizer, device):
    """
    训练一个epoch
    :param epoch:
    :param model:
    :param train_loader:
    :param criterion:
    :param optimizer:
    :param device:
    :return: 该epoch的平均损失
    """
    # 存储每个batch的loss
    loss_history = []

    # 遍历数据集，显示进度条
    total_batch = len(train_loader)  # 便于调用
    for index, data in tqdm(enumerate(train_loader), desc=f"Epoch {epoch}", total=total_batch):
        # 转移数据
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

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

    return sum(loss_history) / total_batch
