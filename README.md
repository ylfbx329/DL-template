# DL模板

本项目旨在为深度学习全流程提供基础代码模板，方便快速构建自己的深度学习项目，使用户能够更专注于模型设计、模型优化等关键问题，减少对通用的代码流程的关注和修改。

## TODO List

- [ ] 将训练和验证日志保存为文件
- [x] 模型验证
- [ ] 训练可视化（TensorBoard、Visdom、**WandB**）

## 使用指南

1. 安装依赖环境
   ```shell
   conda env create -f ./environment.yml
   ```
2. 参考`configs/default.yaml`，根据项目需要，在`configs`下创建配置文件（e.g., `configs/exp.yaml`）
3. 修改`script/main.sh`内容
    - 将变量`PROJECT_ROOT`设置为项目根目录（e.g., `/home/username/DL-template`）
    - 将变量`CONFIG_FILE`设置为使用的配置文件（e.g., `$PROJECT_ROOT/configs/default.yaml`）
4. 运行`script/main.sh`
   ```shell
   bash script/main.sh
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
├── src                    # 源代码文件夹
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
├── script                 # shell脚本文件夹
│   └── main.sh            # 主脚本（项目入口）
├── test                   # 测试文件夹
├── outputs                # 实验结果输出文件夹
│   └── <exp_name>
│       ├── checkpoints
│       ├── logs
│       └── results
├── environment.yml        # conda环境描述文件
└── README.md
```
