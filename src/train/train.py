import matplotlib.pyplot as plt
import torch
import wandb

from src.config.config import Config
from src.data.data_loader import get_data_loader
from src.train.trainer import train_one_epoch
from src.models.net import create_model
from src.criterion.criterion import create_criterion
from src.optim.optim import create_optimizer
from src.utils.utils import save_ckpt, get_transform, get_output_path


def train():
    """
    训练模型的完整流程
    :param param:
    :return:
    """
    print('Start train...')
    # 参数设置
    data_param = Config.args.data
    model_param = Config.args.model
    train_param = Config.args.train

    # 训练设备设置
    device = torch.device(Config.args.device)

    # 定义数据变换操作
    transform = get_transform()

    # 数据加载对象
    train_loader = get_data_loader(data_param.root, train_param.batch_size, split='train', transform=transform)

    # 初始化模型、损失函数和优化器
    model = create_model()
    criterion = create_criterion()
    optimizer = create_optimizer(model, train_param.lr)

    # 设置训练模式
    model.train()

    # 转移模型
    model.to(device)

    # 存储每个epoch的平均loss
    epoch_losses = []

    # 训练模型
    total_epochs = train_param.epochs  # 便于调用
    for epoch in range(total_epochs):
        # 训练一个epoch，获取epoch平均loss
        epoch_loss = train_one_epoch(epoch, model, train_loader, criterion, optimizer, device)
        epoch_losses.append(epoch_loss)

        # 在设定的轮数和训练结束时保存ckpt
        if epoch % train_param.save_freq == 0 or epoch == total_epochs - 1:
            # 保存此轮模型
            ckpt_path = get_output_path(filename=f'epoch{epoch}.pth', type='checkpoint')
            save_ckpt(ckpt_path, epoch, model, optimizer, epoch_loss)
            print(f'Save model at {ckpt_path}')

        # 信息输出，可自定义
        print(f'Epoch [{epoch}/{total_epochs}]: Loss: {epoch_loss}')
        if Config.args.use_wandb:
            wandb.log({"loss": epoch_loss})

    # epoch-loss图像，可自定义
    plt.plot(epoch_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xlim([-2, total_epochs + 2])
    plt.savefig(get_output_path(filename='loss.png', type='result'))
    plt.show()
    print('End train!')
