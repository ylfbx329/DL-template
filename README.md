# DL模板

本项目旨在为深度学习全流程提供基础代码模板，方便快速构建自己的深度学习项目，使用户能够更专注于模型设计、模型优化等关键问题，减少对通用的代码流程的关注和修改。

## TODO List

- [ ] 将训练和验证日志保存为文件
- [x] 模型验证
- [x] 训练可视化（TensorBoard、Visdom、**WandB**）
    - [ ] Rethink WandB功能
- [x] 项目空目录追踪
- [ ] 多卡训练支持

## 使用指南

1. 安装依赖环境
   ```shell
   conda env create -f environment.yml
   ```
   【可选】根据`environment.yml`更新依赖环境，`--prune`选项使conda从环境中删除不再需要的任何依赖项
   ```shell
   conda env update --file environment.yml --prune
   ```
2. 参考`configs/default.yaml`，根据项目需要，在`configs`下创建自己的配置文件
3. 训练模型，运行`src/main.py`脚本
   ```shell
   python -m src.main ./configs/<your-config>.yaml --is_train
   ```
4. 预测模型，运行`src/main.py`脚本
   ```shell
   python -m src.main ./configs/<your-config>.yaml --is_eval
   ```

## 项目目录

```
DL-template                # 项目根目录
├── configs                # 配置文件夹
│   └── default.yaml       # 默认配置文件
├── data                   # 数据文件夹
│   ├── raw                # 原始数据
│   └── processed          # 处理后数据
├── notebooks              # Jupyter文件夹
├── projects               # 相关项目文件夹
├── src                    # 源代码文件夹
│   ├── config             # 全局配置
│   │   └── config.py
│   ├── data               # 数据
│   │   ├── dataset.py     # 数据集
│   │   └── data_loader.py # 数据加载器
│   ├── models             # 模型
│   │   └── net.py         # 主模型
│   ├── optim              # 优化器
│   │   └── optim.py
│   ├── criterion          # 损失函数
│   │   └── criterion.py
│   ├── train              # 训练
│   │   ├── train.py       # 训练主脚本
│   │   └── trainer.py     # 单轮训练
│   ├── evaluate           # 验证
│   │   └── evaluate.py
│   └── utils              # 工具文件夹
│       └── utils.py
├── test                   # 测试文件夹
├── outputs                # 实验结果输出文件夹
│   └── <exp_name>
│       ├── checkpoint
│       ├── log
│       └── result
├── environment.yml        # conda环境描述文件
└── README.md
```
