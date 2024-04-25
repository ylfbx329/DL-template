import argparse
import os
import pprint
import time

import numpy as np
import torch.optim as optim
import torchvision
import yaml
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms

from models.alexnet import *


def read_cfg(cfg_path):
    with open(cfg_path, 'r') as file:
        param = yaml.safe_load(file)
    param['best_test_loss'] = float('inf')
    param['best_train_loss'] = float('inf')
    param['exp_path'] = os.path.join(param['exp_dir'], param['exp_name'])
    param['model_path'] = os.path.join(param['exp_path'], 'model')
    return param


# 设置数据预处理和加载
transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32),
    transforms.Pad((224 - 28) // 2)
])


def train(param):
    pprint.pprint(param)
    device = torch.device(param['device'])

    train_set = torchvision.datasets.MNIST(root=param['data_root'], train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=param['train_batch_size'], shuffle=True, num_workers=0)

    # 初始化模型、损失函数和优化器
    model = AlexNet(in_channels=param['in_channels'], num_classes=param['num_classes'])
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=param['train_lr'])
    model.to(device)
    loss.to(device)

    loss_history = []
    epoch_time = []

    os.makedirs(param['model_path'], exist_ok=True)

    # 训练模型
    for epoch in range(1, param['train_max_epoch'] + 1):
        epoch_loss = []
        start = time.time()
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            batch_loss = loss(outputs, labels)
            batch_loss.backward()
            optimizer.step()

            epoch_loss.append(batch_loss.item())
            # 大数据集时启用
            if (i + 1) % (len(train_loader) // 10) == 0:
                print('Epoch {}: batch[{}/{}] loss: {:.3f}'.format(epoch, i + 1, len(train_loader), epoch_loss[-1]), end='\r')

        loss_history.append(np.mean(epoch_loss))

        model_path = os.path.join(param['model_path'], 'ep{}.pth'.format(epoch))
        torch.save(model, model_path)

        end = time.time()
        epoch_time.append(end - start)
        sum_time = sum(epoch_time)
        etc_time = sum_time / len(epoch_time) * (param['train_max_epoch'] - epoch)
        minutes = int(sum_time // 60)
        seconds = int(sum_time % 60)
        eta_minutes = int(etc_time // 60)
        ets_seconds = int(etc_time % 60)
        print('Epoch {}/{}: loss: {:.3f} ETA: {:02d}:{:02d}/{:02d}:{:02d} Save model at {}'
              .format(epoch, param['train_max_epoch'], loss_history[-1], minutes, seconds, eta_minutes, ets_seconds, model_path))

        if loss_history[-1] < param['best_train_loss']:
            param['best_train_loss'] = loss_history[-1]
            model_path = os.path.join(param['model_path'], 'best_loss.pth')
            torch.save(model, model_path)
            print('Epoch {}: Save best train model at {}'.format(epoch, model_path))

        if epoch % param['val_freq'] == 0:
            val(param, epoch)

    plt.plot(range(1, len(loss_history) + 1), loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(param['exp_path'], 'loss.png'))
    plt.show()
    print('Finished Training')


def val(param, epoch=None):
    device = torch.device(param['device'])

    test_set = torchvision.datasets.MNIST(root=param['data_root'], train=False, download=True, transform=transform)
    val_loader = DataLoader(test_set, batch_size=param['val_batch_size'], shuffle=True, num_workers=0)

    if param['val']:
        model = torch.load(os.path.join(param['model_path'], 'best.pth'))
    else:
        model = torch.load(os.path.join(param['model_path'], 'ep{}.pth'.format(epoch) if epoch else 'best_loss.pth'))
    model.to(device)
    model.eval()

    loss = nn.CrossEntropyLoss()
    loss.to(device)

    loss_history = []
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            batch_loss = loss(outputs, labels)
            loss_history.append(batch_loss.item())

    if param['train'] and np.mean(loss_history) < param['best_test_loss']:
        param['best_test_loss'] = np.mean(loss_history)
        model_path = os.path.join(param['model_path'], 'best.pth')
        torch.save(model, model_path)

    print('Accuracy of the network on the {} test samples: {:.3f}%'.format(len(test_set), np.mean(loss_history)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='./cfg/default.yaml')
    parser.add_argument('--train', action="store_true", default=True)
    parser.add_argument('--val', action="store_true", default=False)
    args = parser.parse_args()

    param = read_cfg(args.cfg)
    param.update(vars(args))

    if args.train:
        train(param)
    if args.val:
        val(param)
