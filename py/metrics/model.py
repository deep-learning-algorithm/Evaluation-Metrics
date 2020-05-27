# -*- coding: utf-8 -*-

"""
@date: 2020/5/25 下午2:01
@file: model.py
@author: zj
@description: 
"""

import time

import torch
import torch.nn as nn
from thop import profile


def compute_gflops_and_model_size(model):
    input = torch.randn(1, 3, 224, 224)
    macs, params = profile(model, inputs=(input,), verbose=False)

    GFlops = macs * 2.0 / pow(10, 9)
    model_size = params * 4.0 / 1024 / 1024
    return GFlops, model_size


def compute_params(model):
    assert isinstance(model, nn.Module)
    return sum([param.numel() for param in model.parameters()])


@torch.no_grad()
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

    return total_time / epoch
