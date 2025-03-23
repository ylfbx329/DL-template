import importlib
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import torchvision
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torchinfo import summary
from torchvision.models import ResNet18_Weights
from tqdm import tqdm

from src.config.config import Config
from src.data.data_loader import get_dataloader
from src.metrics.metrics import metrics
from src.utils.utils import save_ckpt, load_ckpt, get_output_path
from src.visualize.visualize import plot


class BaseExp:
    def __init__(self):
        super(BaseExp, self).__init__()
        self.device = torch.device(Config.args.device)

        self.train_dataloader, self.val_dataloader, self.test_dataloader = get_dataloader()
        self.model = self.build_model().to(self.device)
        self.criterion = self.get_criterion()
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler()

    def _load_class(self,
                    module_name_list: str | list[str],
                    params: SimpleNamespace,
                    exclude_keys: Optional[list[str]] = None):
        """
        动态加载模块并实例化指定类对象
        类由params.name指定
        :param module_name_list: 动态加载的模块列表
        :param params: 实例化类对象所需的参数字典
        :param exclude_keys: 排除参数字典中的与实例化无关的字段
        :return: 指定类的实例化对象
        """
        if not isinstance(module_name_list, list):
            module_name_list = [module_name_list]
        if exclude_keys is None:
            exclude_keys = ['name']

        cls = None
        for module_name in module_name_list:
            # 动态加载模块
            module = importlib.import_module(module_name)
            # 获取模块内的指定类
            if hasattr(module, params.name):
                cls = getattr(module, params.name)
                break

        if cls is None:
            raise ValueError(f'Class "{params.name}" not found. Please ensure the class is defined in {module_name_list}.')

        cls_param = {key: value
                     for key, value in Config.get_argsdict(params).items()
                     if key not in exclude_keys}

        # 实例化类对象
        obj = cls(**cls_param)
        return obj

    def build_model(self) -> nn.Module:
        """
        创建模型
        :return: 模型对象
        """
        model_param = Config.args.model
        model = self._load_class('src.models.net', model_param)

        # 测试专用 start
        model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        model.conv1 = nn.Conv2d(model_param.in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(512, model_param.num_classes)
        # 测试专用 end

        # 打印类信息，防止错误构建
        logging.info(f'create model: {model.__class__}')
        return model

    def get_criterion(self) -> nn.Module:
        """
        构建损失函数
        :return: 损失函数对象
        """
        loss_param = Config.args.loss
        criterion = self._load_class(['src.criterion.criterion', 'torch.nn'], loss_param)

        # 打印类信息，防止错误构建
        logging.info(f'create criterion: {criterion.__class__}')
        return criterion

    def get_optimizer(self):
        """
        构建优化器
        :return: 优化器对象
        """
        optim_param = Config.args.optim
        # 构建优化器需要模型参数
        optim_param.params = self.model.parameters()
        optimizer = self._load_class('torch.optim', optim_param)

        # 打印类信息，防止错误构建
        logging.info(f'create optimizer: {optimizer.__class__}')
        return optimizer

    def get_scheduler(self):
        """
        构建调度器
        配置文件中无调度器设置则采用恒定调度器，与不采用调度器效果等同
        :return: 调度器对象
        """
        if hasattr(Config.args, 'sched'):
            sched_param = Config.args.sched
            # 构建调度器需要优化器作为参数
            sched_param.optimizer = self.optimizer
            scheduler = self._load_class('torch.optim.lr_scheduler', sched_param)
        else:
            # 默认调度器不会改变学习率，用于方便设计统一的训练流程
            scheduler = LambdaLR(self.optimizer, lr_lambda=lambda _: 1.0)

        # 打印类信息，防止错误构建
        logging.info(f'create scheduler: {scheduler.__class__}')
        return scheduler

    def train(self):
        """
        训练模型的通用完整流程
        """
        logging.info('Start train...')
        train_param = Config.args.train
        val_param = Config.args.val

        # 断点续训
        if Config.args.resume is not None:
            load_ckpt(Config.args.resume, self.model, self.optimizer, self.scheduler)

        if Config.args.once:
            # 打印模型结构及参数
            # 此处next(iter(train_loader))并不会干扰第一个epoch的迭代，即第一个epoch仍会完整的处理整个数据集
            # 但此处in_data与第一个epoch的第一个batch数据不同，是由于train_loader的shuffle为true，若为false则相同
            in_data, _ = [x.to(self.device) for x in next(iter(self.train_dataloader))]
            summary(self.model, input_data=in_data)

        # 存储每个epoch的学习率和平均损失
        epoch_lr = []
        epoch_losses = []

        # 记录最好的验证集精度
        best_val_acc = -np.inf

        # 训练模型
        start_epoch = self.scheduler.last_epoch + 1  # 从头训练时start_epoch=1
        total_epochs = train_param.epochs
        for epoch in range(start_epoch, total_epochs + 1):
            # 训练一个epoch，获取epoch平均损失
            epoch_loss = self.train_epoch(epoch)
            epoch_losses.append(epoch_loss)
            epoch_lr.append(self.scheduler.get_last_lr())

            # 调度器更新，必须在ckpt保存之前，否则调度器的轮次记录与实际训练轮次不符，导致断点续训错误
            self.scheduler.step()

            # 每个epoch结束后的信息输出，可自定义
            logging.info(f'Epoch [{epoch}/{total_epochs}]: lr: {epoch_lr[-1]}, Loss: {epoch_loss}')

            # 在设定的轮数和训练结束时保存ckpt
            if epoch % train_param.save_epoch == 0 or epoch == total_epochs:
                ckpt_filename = f'epoch{epoch}.pth'
                save_ckpt(ckpt_filename, self.model, self.optimizer, self.scheduler, epoch, epoch_loss)

            # 在设定的轮数和训练结束时使用验证集验证模型
            if epoch >= train_param.val_start and (epoch % train_param.val_epoch == 0 or epoch == total_epochs):
                # 模型验证，返回模型输出、预测结果、真值标签、验证集平均损失、验证集精度
                val_output, val_result, val_label, val_loss, val_accuracy = self.evaluate(self.val_dataloader, val_param.log_iter)
                # 模型验证信息输出，可自定义
                logging.info(f'Validate: Epoch: {epoch}, Loss: {val_loss}')
                # 保存在验证集指标最优的模型
                if val_accuracy > best_val_acc:
                    best_val_acc = val_accuracy
                    logging.info(f'Validate: Epoch: {epoch}, Best Accuracy: {val_accuracy}')
                    save_ckpt('best_val.pth', self.model, self.optimizer, self.scheduler, epoch, epoch_loss)

        if total_epochs >= start_epoch:
            # 每个epoch的损失折线图，可自定义
            plot(x=range(start_epoch, total_epochs + 1),
                 y=epoch_losses,
                 xlabel='Epoch',
                 ylabel='Loss',
                 image_filename=f'epoch{start_epoch}-{total_epochs}_loss.png')
            # 每个epoch的学习率折线图，可自定义
            plot(x=range(start_epoch, total_epochs + 1),
                 y=epoch_lr,
                 xlabel='Epoch',
                 ylabel='Learning Rate',
                 image_filename=f'epoch{start_epoch}-{total_epochs}_lr.png')

        logging.info('End train!')

    def train_epoch(self, epoch):
        """
        训练一个epoch
        :param epoch: 第epoch轮训练
        :return: 本轮次平均损失
        """
        # 设置模型为训练模式
        self.model.train()

        # 存储每个batch的损失
        batch_losses = []

        total_batch = len(self.train_dataloader)
        log_iter = Config.args.train.log_iter
        # 遍历数据集，显示进度条
        for index, data in tqdm(enumerate(self.train_dataloader), desc=f"Epoch {epoch}", total=total_batch):
            # 转移数据
            inputs, labels = [x.to(self.device) for x in data]

            # 清零梯度
            self.optimizer.zero_grad()

            # 前向传播
            outputs = self.model(inputs)

            # 计算损失
            loss = self.criterion(outputs, labels)

            # 反向传播
            loss.backward()

            # 参数更新
            self.optimizer.step()

            # 记录损失
            batch_losses.append(loss.item())

            # 日志打印
            if log_iter > 0 and index % log_iter == 0:
                logging.info(f'Batch [{index}/{total_batch}]: mean loss: {np.mean(batch_losses)}')

        return np.mean(batch_losses)

    @torch.no_grad()
    def evaluate(self, dataloader, log_iter):
        """
        评估模型性能的通用流程
        :param dataloader: 评估使用的数据集（验证集或测试集）的dataloader
        :param log_iter: 在一轮评估过程中打印日志的batch频率
        :return: 模型输出、预测结果、真值标签、平均损失、精度
        """
        # 设置模型为评估模式
        self.model.eval()

        # 存储每个batch的模型输出、预测结果、真值标签、损失
        output_list = []
        result_list = []
        label_list = []
        batch_losses = []

        total_batch = len(dataloader)
        # 遍历数据集，显示进度条
        for index, data in tqdm(enumerate(dataloader), desc="Evaluate", total=total_batch):
            # 转移数据
            inputs, labels = [x.to(self.device) for x in data]

            # 前向传播
            outputs = self.model(inputs)

            # 计算损失
            loss = self.criterion(outputs, labels)

            # 计算预测结果，可自定义
            result = torch.argmax(outputs.detach(), dim=1)

            # 记录模型输出
            output_list.append(outputs.detach().cpu().numpy())

            # 记录预测结果
            result_list.append(result.detach().cpu().numpy())

            # 记录真值标签
            label_list.append(labels.detach().cpu().numpy())

            # 记录损失
            batch_losses.append(loss.item())

            # 日志打印
            if log_iter > 0 and index % log_iter == 0:
                logging.info(f'Batch [{index}/{total_batch}]: mean loss: {np.mean(batch_losses)}')

        # 重整为(batch,)
        output = np.concatenate(output_list)
        result = np.concatenate(result_list)
        label = np.concatenate(label_list)
        # 计算平均损失
        avg_loss = np.mean(batch_losses)

        # 计算指标
        accuracy = metrics(label, result)
        return output, result, label, avg_loss, accuracy

    @torch.no_grad()
    def validate(self):
        """
        使用验证集评估模型
        :return:
        """
        logging.info('Start validate...')
        val_param = Config.args.val
        # 加载ckpt
        load_ckpt(val_param.ckpt, self.model)
        output, result, label, loss, accuracy = self.evaluate(self.val_dataloader, val_param.log_iter)

        # 信息输出，可自定义
        logging.info(f'Validate: Loss: {loss}')
        logging.info('End validate!')

    @torch.no_grad()
    def test(self):
        """
        使用测试集评估模型
        :return:
        """
        logging.info('Start test...')
        test_param = Config.args.test
        # 加载ckpt
        load_ckpt(test_param.ckpt, self.model)
        output, result, label, loss, accuracy = self.evaluate(self.test_dataloader, test_param.log_iter)

        # 信息输出，可自定义
        logging.info(f'Test: Loss: {loss}')

        # 保存测试结果
        ckpt = Path(test_param.ckpt).stem
        result_path = get_output_path(filename=f'{ckpt}-result.npy', filetype='result')
        label_path = get_output_path(filename=f'{ckpt}-label.npy', filetype='result')
        np.save(result_path, result)
        np.save(label_path, label)
        logging.info(f'Save result at {result_path} and label at {label_path}')

        logging.info('End test!')
