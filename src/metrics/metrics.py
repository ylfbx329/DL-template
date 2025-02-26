import logging

from sklearn.metrics import accuracy_score


def metrics(y_true, y_pred):
    """
    计算指标、打印为日志并返回指标结果
    :param y_true: 真值标签
    :param y_pred: 模型输出结果
    :return: 各项指标
    """
    # 自行补充其余指标
    accuracy = accuracy_score(y_true, y_pred)

    # 打印指标
    logging.info(f"Accuracy: {accuracy}")
    return accuracy
