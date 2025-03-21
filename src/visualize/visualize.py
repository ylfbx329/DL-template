from matplotlib import pyplot as plt

from src.utils.utils import get_output_path


def plot(x, y, xlabel, ylabel, image_filename):
    """
    绘制简单的折线图
    :param x: 折线点的x轴坐标列表
    :param y: 折线点的y轴坐标列表
    :param xlabel: x轴标签
    :param ylabel: y轴标签
    :param image_filename: 用于保存的图像文件名
    """
    # 使图像紧凑，避免多余白边
    plt.figure(layout='constrained')
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    title = image_filename.rsplit('.', 1)[0]
    plt.title(title)
    # 避免折线贴边，使图像美观
    plt.xlim([x[0] - 2, x[-1] + 2])
    plt.savefig(get_output_path(filename=image_filename, filetype='result'))
    plt.show()
