# DL模板

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
    - 将`PROJECT_ROOT`设置为项目根目录（e.g., `/home/username/DL-template`）
    - 将`CONFIG_FILE`设置为使用的配置文件（e.g., `$PROJECT_ROOT/configs/default.yaml`）
4. 运行`script/main.sh`
   ```shell
   bash script/main.sh
   ```
