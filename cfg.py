import yaml

param = dict()


def read_cfg(cfg_path):
    global param
    with open(cfg_path, 'r') as file:
        param = yaml.safe_load(file)
