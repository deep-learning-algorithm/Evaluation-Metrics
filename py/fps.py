# -*- coding: utf-8 -*-

"""
@date: 2020/4/28 下午2:10
@file: fps.py
@author: zj
@description: 计算模型实现速度 - FPS
"""

import time
import torch
from torchvision.models import alexnet

from utils import util


def compute_fps(model, shape, epoch=100, device=None):
    """
    frames per second
    :param shape: 输入数据大小
    """
    total_time = 0.0

    if device:
        model = model.to(device)
    for i in range(epoch):
        data = torch.randn(shape)
        if device:
            data = data.to(device)

        start = time.time()
        outputs = model(data)
        end = time.time()

        total_time += (end - start)

    return epoch / total_time


if __name__ == '__main__':
    num_classes = 10
    model = alexnet(num_classes=num_classes)

    device = util.get_device()
    fps = compute_fps(model, (1, 3, 224, 224), device=device)
    print('fps: {:.3f}'.format(fps))
