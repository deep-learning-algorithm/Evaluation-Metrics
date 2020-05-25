# -*- coding: utf-8 -*-

"""
@date: 2020/5/25 下午2:01
@file: model.py
@author: zj
@description: 
"""

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
