# 深度学习模板

本项目旨在为深度学习全流程提供基础代码模板，方便快速构建自己的深度学习项目，使用户能够更专注于模型设计、模型优化等关键问题，减少对通用的代码流程的关注和修改。

## 功能特性

- 通用模板：实现了**数据加载 -> 模型创建 -> 训练优化 -> 模型验证 -> 模型测试**完整流程
- 配置为王：仅修改YAML配置文件即可控制模型、损失函数、优化器和调度器的所有参数，无需改动代码
- 实验管理：通过集成`BaseExp`类，快速开发预训练、对比实验等相似实验逻辑
- 实验隔离：通过YAML配置文件控制不同实验结果的隔离，高效对比实验结果
- 新手友好：包含大量详细注释，便于新手理解并创建自己的深度学习程序
- 随处访问：将命令行参数和YAML配置文件统一封装到Config类，可在任意位置获取任一参数，一次创建，随处获取
- 日志管理：详细的Python原生日志记录，在文件和控制台中同步输出日志信息

## 未来开发计划

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
    - [ ] 早停策略配置（当前val_loss最低则停不恰当）
- 日志
    - [x] 日志持久化保存
    - [x] 轮内日志打印配置
- 其他
    - [x] 项目空目录追踪

## 使用说明

### 运行实验

1. 创建conda环境：根据项目需求，修改`environment.yml`环境配置文件，并创建环境
    ```shell
    conda env create -f environment.yml
    ```
   【注】项目需求变动，可直接更改`environment.yml`后运行以下命令更新环境，`--prune`选项表示从环境中删除不再需要的任何依赖项
    ```shell
    conda env update --file environment.yml --prune
    ```
2. 适配开发
    - 配置文件：参考`configs/default.yaml`，根据项目需要，在`configs`下创建自己的配置文件（e.g., `configs/mnist.yaml`，则`mnist`被设置为实验名（`<exp_name>`））
    - 数据
        - 在`src/data/data_loader.py`中实现数据加载函数，返回dataloader
        - 在`src/data/dataset.py`中实现数据集类
    - 模型：在`src/models/net.py`中实现主模型
    - 损失函数：在`src/criterion/criterion.py`中实现自定义的损失函数
    - 实验：继承`src/exp/base_exp.py`中的`BaseExp`类，实现自定义的模型等构建逻辑，以及训练、验证和测试逻辑。可便捷实现预训练、对比实验等实验逻辑
    - 评价指标：在`src/metrics/metrics.py`添加所需评价指标的计算
    - 可视化：在`src/visualize/visualize.py`添加所需的可视化函数
3. 命令行启动参数解释
    - `-cfg <path/to/cfg>`指定本次实验使用的配置文件
    - `-train`、`-val`和`test`运行训练、验证和测试任务，可同时指定，将按照训练-验证-测试的顺序依次执行
    - `-resume <ckpt_filename>`断点续训模式，指定继续训练使用的检查点，`<ckpt_filename>`需以`.pth`结尾，将加载`outputs/<cfg_filename>/checkpoint/<ckpt_filename>`
        - `-resume`不可用于修改epoch后的延长训练，会导致学习率调度器不按预期更新学习率
4. 启动命令示例，`-m`选项以模块方式运行，用于自动解决引用路径问题
    ```shell
    cd <project_root> # 切换到项目根目录下
    python -m src.main -cfg ./configs/default.yaml -train # 训练模型
    python -m src.main -cfg ./configs/default.yaml -train -resume epoch3.pth # 断点续训
    python -m src.main -cfg ./configs/default.yaml -val # 验证模型
    python -m src.main -cfg ./configs/default.yaml -test # 测试模型
    python -m src.main -cfg ./configs/default.yaml -train -test # 训练并测试模型
    ```
5. 模型检查点、日志文件和模型输出结果保存在`outputs/<exp_name>`下（e.g., `outputs/mnist`）

## 进阶技巧

1. 使用Git子模块管理参考项目
    - 克隆参考项目
        ```shell
        git submodule add --depth=1 <repo_url> <path/to/submodule>
        # 以本项目举例
        git submodule add --depth=1 https://github.com/ylfbx329/DL-template.git projects/DL-template
        ```
    - 更新项目`git submodule update --remote --merge`
2. 使用Git分支管理不同模型版本
    - 新建分支`git checkout -b new_net`
    - 开发新版本模型
    - 提交`git commit -m "new_net 模型实现"`
    - 上传服务器，运行实验
    - 若要此时开发另一新版本模型（若不修改代码则可新建配置文件后直接再次运行程序）
        - 同步服务器代码`rsync -av --exclude=outputs/ ~/project/ ~/new_project/`
            - `-a`递归同步并完全保持源文件的属性，`-v`详细输出
            - `--exclude`指定不同步的目录，可有多个，即`--exclude=aaa --exclude=bbb`
        - 调整PyCharm部署路径到`~/new_project/`
        - 签出到基于开发的分支`git checkout base`
        - 新建分支、开发、提交、上传服务器、运行实验
    - 记录实验结果在当前分支的README.md中并提交`git commit -m "new_net 结果记录"`，同时记录一份在与项目隔离的文件中
    - 若实验结果有效
        - 切换到主分支`git checkout master`
        - 合并分支`git merge new_net`
    - 若实验结果无效
        - 切换到主分支`git checkout master`
        - 从git日志中选择需要合并的提交进行优选，详见[PyCharm文档](https://www.jetbrains.com.cn/en-us/help/pycharm/apply-changes-from-one-branch-to-another.html#apply-separate-changes)
    - 注：切换分支时若有未提交的修改，使用PyCharm提供的搁置（shelve）或Git的隐藏（stash）功能暂存代码，切换回来时恢复

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
├── script                  # Shell脚本文件夹
├── src                     # 源代码文件夹
│   ├── config              # 配置
│   │   └── config.py       # 配置类
│   ├── data                # 数据
│   │   ├── utils.py        # 数据工具
│   │   ├── dataset.py      # 实现数据集类
│   │   └── data_loader.py  # 实现数据加载函数
│   ├── models              # 模型
│   │   └── net.py          # 实现主模型
│   ├── criterion           # 损失函数
│   │   └── criterion.py    # 实现自定义损失函数
│   ├── exp                 # 实验逻辑
│   │   ├── base_exp.py     # 基础通用实验逻辑
│   │   ├── <custom_exp.py> # 预训练、对比方法等自定义实验逻辑
│   │   └── ...
│   ├── metrics             # 评价指标
│   │   └── metrics.py
│   ├── visualize           # 可视化
│   │   └── visualize.py
│   ├── utils               # 通用工具
│   │   └── utils.py
│   └── main.py             # 项目主脚本
├── test                    # 测试文件夹
├── outputs                 # 实验结果输出文件夹
│   └── <exp_name>
│       ├── checkpoint      # 模型检查点
│       ├── log             # 运行日志
│       └── result          # 运行结果
├── .gitignore              # Git忽略文件
├── environment.yml         # conda环境描述文件
└── README.md
```
