# 深度学习模板

本项目旨在为深度学习全流程提供基础代码模板，方便快速构建自己的深度学习项目，使用户能够更专注于模型设计、模型优化等关键问题，减少对通用的代码流程的关注和修改。

## Features

- 通用模板：实现了`数据加载 -> 模型创建 -> 训练优化 -> 模型验证 -> 模型测试`完整流程
- 新手友好：包含大量详细注释，便于新手理解并创建自己的深度学习程序
- 超参管理：通过YAML文件统一结构化管理超参，使用Config类一次创建，到处获取
- 日志管理：详细的Python原生日志记录，在文件和控制台中同步输出日志信息
- 无感调度：实现了无感的调度器逻辑，方便新手默认和老手自定义
- 对比实验：通过YAML配置文件控制不同实验结果的隔离，高效对比实验结果

## TODO List

- 通用
    - [x] 模型测试逻辑
    - [x] 模型验证逻辑及最优模型保存
    - [x] ckpt加载配置
    - [x] 随机种子配置
    - [ ] 多卡支持
    - [ ] 实验可视化（TensorBoard、Visdom、**WandB**）
- 训练
    - [x] 调度器配置
    - [x] 模型结构统计
    - [ ] 早停策略配置
- 日志
    - [x] 日志持久化保存
    - [x] 轮内日志打印配置
- 其他
    - [x] 项目空目录追踪

## Usage

1. 根据需要修改`environment.yml`文件，创建conda环境
   ```shell
   conda env create -f environment.yml
   ```
   【可选】根据`environment.yml`更新依赖环境，`--prune`选项表示从环境中删除不再需要的任何依赖项
   ```shell
   conda env update --file environment.yml --prune
   ```
2. 参考`configs/default.yaml`，根据项目需要，在`configs`下创建自己的配置文件（e.g., `configs/mnist.yaml`，则`mnist`被设置为实验名（<exp_name>）以下命令以此配置文件为例）
3. 指定运行模式（训练、验证、测试），可指定多种模式，`-m`选项用于自动解决引用路径问题
   ```shell
   cd <project_root> # 切换到项目根目录下
   python -m src.main -cfg ./configs/mnist.yaml -train # 训练模型
   python -m src.main -cfg ./configs/mnist.yaml -val # 验证模型
   python -m src.main -cfg ./configs/mnist.yaml -test # 测试模型
   python -m src.main -cfg ./configs/mnist.yaml -train -test # 训练并测试模型
   ```
4. 模型检查点、日志文件和模型输出结果保存在`outputs/mnist`下

## 项目目录

```
DL-template                 # 项目根目录
├── configs                 # 配置文件夹
│   └── default.yaml        # 默认配置文件
├── data                    # 数据文件夹
│   ├── raw                 # 原始数据
│   └── processed           # 预处理后数据
├── notebooks               # Jupyter文件夹
├── projects                # 相关项目文件夹
├── src                     # 源代码文件夹
│   ├── config              # 配置
│   │   └── config.py       # 全局配置
│   ├── data                # 数据
│   │   ├── dataset.py      # 数据集
│   │   └── data_loader.py  # 数据加载器
│   ├── models              # 模型
│   │   └── net.py          # 主模型
│   ├── criterion           # 损失函数
│   │   └── criterion.py
│   ├── optim               # 优化器和调度器
│   │   └── optim.py
│   ├── train               # 训练
│   │   ├── train.py        # 训练主脚本
│   │   └── trainer.py      # 单轮训练
│   ├── validate            # 验证
│   │   ├── validate.py     # 验证主脚本
│   │   └── validator.py    # 单轮验证
│   ├── test                # 测试
│   │   ├── test.py         # 测试主脚本
│   │   └── tester.py       # 单轮测试
│   ├── metrics             # 评估指标
│   │   └── metrics.py
│   ├── visualize           # 可视化
│   │   └── visualize.py
│   ├── utils               # 通用工具
│   │   └── utils.py
│   └── main.py             # 项目主脚本
├── test                    # 测试文件夹
├── outputs                 # 实验结果输出文件夹
│   └── <exp_name>
│       ├── checkpoint
│       ├── log
│       └── result
├── environment.yml         # conda环境描述文件
└── README.md
```
