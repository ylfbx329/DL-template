import logging
import pprint
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Union


class Config:
    """
    配置类，便于全局访问参数
    """
    args = SimpleNamespace()

    @classmethod
    def update_args(cls, new_args: Union[Dict, List[Dict]], src_namespace: SimpleNamespace = None):
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
        if src_namespace is None:
            src_namespace = cls.args
        for param_dict in new_args:
            _recursive_update(src_namespace, param_dict)

    @classmethod
    def get_argsdict(cls, namespace: SimpleNamespace = None) -> Dict:
        """
        将namespace递归解析为字典
        :param namespace: 待解析的namespace
        :return: 解析出的参数字典
        """
        if namespace is None:
            namespace = cls.args
        return {key: cls.get_argsdict(value) if isinstance(value, SimpleNamespace) else value
                for key, value in vars(namespace).items()}

    @classmethod
    def logging_args(cls):
        """
        整齐打印Config.args为日志
        """
        # 单独处理Path类，将其转换为字符串，避免打印多余内容
        args_str = pprint.pformat({key: str(value) if isinstance(value, Path) else value
                                   for key, value in cls.get_argsdict().items()})
        logging.info(args_str)
