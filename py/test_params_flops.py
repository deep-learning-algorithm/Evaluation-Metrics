# -*- coding: utf-8 -*-

"""
@date: 2020/5/25 下午2:02
@file: test_params_flops.py
@author: zj
@description: 
"""

from torchvision.models import AlexNet

from metrics.model import compute_params
from metrics.model import compute_gflops_and_model_size


def test_model_param_num():
    """
    计算模型参数个数，同时返回相应的大小（MB）
    size = num * 4.0 / 1024 / 1024
    """
    model = AlexNet()
    num_params = compute_params(model)
    size_params = num_params * 4.0 / 1024 / 1024

    print('num_params: {} - size_params: {:.3f} MB'.format(num_params, size_params))
    return num_params, size_params


def test_model_flops():
    """
    计算模型GFLops，同时返回模型大小（MB）
    """
    model = AlexNet()
    gflops, model_size = compute_gflops_and_model_size(model)

    print('{:.3f} GFlops - {:.3f} MB'.format(gflops, model_size))


if __name__ == '__main__':
    test_model_param_num()
    test_model_flops()
