import pprint
from types import SimpleNamespace
from typing import Dict, List, Union


class Config:
    """
    配置类，便于全局访问参数
    """
    args = SimpleNamespace()

    @staticmethod
    def update_args(new_args: Union[Dict, List[Dict]]) -> None:
        """
        更新Config.args
        :param new_args:
        :return:
        """
        if isinstance(new_args, dict):
            new_args = [new_args]
        for param_dict in new_args:
            Config._recursive_update(Config.args, param_dict)

    @staticmethod
    def _recursive_update(namespace, updates):
        """
        将参数字典递归更新到namespace中
        :param namespace: 待更新的namespace
        :param updates: 参数字典
        :return:
        """
        for key, value in updates.items():
            if isinstance(value, dict):
                current_attr = getattr(namespace, key, SimpleNamespace())
                Config._recursive_update(current_attr, value)
                setattr(namespace, key, current_attr)
            else:
                setattr(namespace, key, value)

    @staticmethod
    def print_args():
        """
        整齐打印Config.args
        :return:
        """

        def namespace_to_dict(namespace):
            """
            将namespace递归解析为字典
            :param namespace:
            :return:
            """
            return {key: namespace_to_dict(value) if isinstance(value, SimpleNamespace) else value
                    for key, value in vars(namespace).items()}

        pprint.pprint(namespace_to_dict(Config.args))
