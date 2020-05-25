# -*- coding: utf-8 -*-

"""
@date: 2020/5/25 下午2:39
@file: test_fps.py
@author: zj
@description: 
"""

from torchvision.models import alexnet
import torch

from metrics.model import compute_fps


def get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    num_classes = 10
    model = alexnet(num_classes=num_classes)

    device = get_device()
    fps = compute_fps(model, (1, 3, 224, 224), device=device)
    print('device: {} - fps: {:.3f}s'.format(device.type, fps))
