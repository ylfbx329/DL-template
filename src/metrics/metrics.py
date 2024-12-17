import logging

from sklearn.metrics import accuracy_score


def metrics(y_true, y_pred):
    """
    计算指标并打印为日志
    :param y_true: 真值标签
    :param y_pred: 模型输出结果
    """
    accuracy = accuracy_score(y_true, y_pred)
    logging.info(f"Accuracy: {accuracy}")
