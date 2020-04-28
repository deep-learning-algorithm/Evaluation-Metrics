# -*- coding: utf-8 -*-

"""
@date: 2020/4/28 上午9:29
@file: flops-params.py
@author: zj
@description: 计算GFlops和模型大小
"""

import torch
from thop import profile
from torchvision.models import AlexNet


def compute_num_flops(model):
    input = torch.randn(1, 3, 224, 224)
    macs, params = profile(model, inputs=(input,), verbose=False)
    # print(macs, params)

    GFlops = macs * 2.0 / pow(10, 9)
    params_size = params * 4.0 / 1024 / 1024
    return GFlops, params_size


if __name__ == '__main__':
    model = AlexNet()
    gflops, params_size = compute_num_flops(model)
    print('{:.3f} GFlops - {:.3f} MB'.format(gflops, params_size))
