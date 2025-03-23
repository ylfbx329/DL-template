import random

import numpy as np
import torch


def seed_worker(worker_id):
    """
    设定每个子进程的随机种子，保证实验的可重复性
    :param worker_id: 子进程ID
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
