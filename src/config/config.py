import logging
import pprint
from types import SimpleNamespace
from typing import Dict, List, Union


class Config:
    """
    配置类，便于全局访问参数
    """
    args = SimpleNamespace()

    @staticmethod
    def update_args(new_args: Union[Dict, List[Dict]], src_namespace: SimpleNamespace = args):
        """
        更新Config.args
        :param new_args: 参数字典
        :param src_namespace: 待更新的namespace，默认为整个args
        """

        def _recursive_update(namespace: SimpleNamespace, updates: Dict):
            """
            将参数字典递归更新到namespace中
            :param namespace: 待更新的namespace
            :param updates: 参数字典
            """
            for key, value in updates.items():
                if isinstance(value, dict):
                    current_attr = getattr(namespace, key, SimpleNamespace())
                    _recursive_update(current_attr, value)
                    setattr(namespace, key, current_attr)
                else:
                    setattr(namespace, key, value)

        if isinstance(new_args, dict):
            new_args = [new_args]
        for param_dict in new_args:
            _recursive_update(src_namespace, param_dict)

    @staticmethod
    def get_argsdict(namespace: SimpleNamespace = args) -> Dict:
        """
        将namespace递归解析为字典
        :param namespace: 待解析的namespace
        :return: 解析出的参数字典
        """
        return {key: Config.get_argsdict(value) if isinstance(value, SimpleNamespace) else value
                for key, value in vars(namespace).items()}

    @staticmethod
    def logging_args():
        """
        整齐打印Config.args为日志
        """
        args_str = pprint.pformat(Config.get_argsdict())
        logging.info(args_str)
