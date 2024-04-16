import yaml

# 读取 YAML 文件
with open('./cfg/default.yaml', 'r') as file:
    config = yaml.safe_load(file)

# 获取配置信息
batch_size = config['train']['batch_size']
cuda = config['exp']['cuda']

print(batch_size)
print(config)
print(1 if cuda else 0)
