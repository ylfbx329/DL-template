#!/bin/bash

# 声明环境变量
PROJECT_ROOT="/home/username/DL-template"
export PROJECT_ROOT
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# 配置文件路径
CONFIG_FILE="$PROJECT_ROOT/configs/default.yaml"

# 运行训练脚本
python "$PROJECT_ROOT/src/train/train.py" $CONFIG_FILE

# 运行测试脚本
python "$PROJECT_ROOT/src/evaluate/evaluate.py" $CONFIG_FILE
