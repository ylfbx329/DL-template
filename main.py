import os
import yaml
import argparse

import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import v2 as transforms
from tqdm import tqdm
from matplotlib import pyplot as plt

from models import *
from loss import *
from cfg import *

# 设置数据预处理和加载
transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32),
    transforms.Pad((224 - 28) // 2)
])

train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)


def train(dataset, param):
    device = torch.device("cuda:{}".format(param['exp']['gpu_id']) if param['exp']['cuda'] and torch.cuda.is_available() else "cpu")
    # train_param = param['train']
    train_loader = DataLoader(dataset, batch_size=param['train']['batch_size'], shuffle=param['train']['shuffle'], num_workers=param['train']['num_workers'])

    # 初始化模型、损失函数和优化器
    model = AlexNet(in_channels=param['dataset']['in_channels'], num_classes=param['dataset']['num_classes'])
    model.apply(init_weights)
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=param['train']['learning rate'])
    model.to(device)
    loss.to(device)

    best_loss = float('inf')
    loss_history = []

    model_dir = os.path.join(param['exp']['exp_dir'], param['exp']['exp_name'], 'model')
    os.makedirs(model_dir, exist_ok=True)

    # 训练模型
    for epoch in tqdm(range(1, param['train']['max_epoch'] + 1)):  # 迭代5次
        print("\n")
        epoch_loss = []
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            batch_loss = loss(outputs, labels)
            batch_loss.backward()
            optimizer.step()

            loss_value = batch_loss.item()

            epoch_loss.append(loss_value)
            if (i + 1) % param['train']['loss_display_freq'] == 0:  # 每2000个mini-batches打印一次损失值
                print('Epoch {}: batch[{}/{}] loss: {:.3f}'.format(epoch, i + 1, len(train_loader), loss_value), end='\r')

        loss_mean = np.mean(epoch_loss)
        loss_history.append(loss_mean)
        print('\nEpoch {}: loss: {:.3f}'.format(epoch, loss_mean))

        if loss_mean < best_loss:
            best_loss = loss_mean
            model_path = os.path.join(model_dir, 'best_loss.pth')
            torch.save(model, model_path)
            print('Epoch {}: Save best model at {}'.format(epoch, model_path))

        if epoch % param['train']['val_epoch_freq'] == 0:
            model_path = os.path.join(model_dir, 'ep{}.pth'.format(epoch))
            torch.save(model, model_path)
            print('Epoch {}: Save model at {}'.format(epoch, model_path))
    plt.plot(range(1, len(loss_history) + 1), loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    print('Finished Training')


def val(dataset, param):
    device = torch.device("cuda:{}".format(param['exp']['gpu_id']) if param['exp']['cuda'] and torch.cuda.is_available() else "cpu")

    val_loader = DataLoader(dataset, batch_size=param['val']['batch_size'], num_workers=param['val']['num_workers'])

    model = torch.load(os.path.join(param['exp']['exp_dir'], param['exp']['exp_name'], 'model', 'best_loss.pth'))
    model.to(device)
    model.eval()

    correct = 0
    with torch.no_grad():
        for i, data in enumerate(tqdm(val_loader)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the {} test images: {:.3f}%'.format(len(dataset), 100 * correct / len(dataset)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='./cfg/default.yaml')
    parser.add_argument('--train', action="store_true", default=False)
    parser.add_argument('--val', action="store_true", default=False)
    args = parser.parse_args()

    read_cfg(args.cfg)

    if args.train:
        train(train_set, param)
    if args.val:
        val(test_set, param)
