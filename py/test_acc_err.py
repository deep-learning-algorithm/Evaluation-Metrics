# -*- coding: utf-8 -*-

"""
@date: 2020/5/25 下午2:56
@file: test_acc_err.py.py
@author: zj
@description: 
"""

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torchvision.models import alexnet

from metrics.acc import compute_accuracy


def get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def test_top1_top5_acc(data_loader, model, device):
    epoch_top1_acc, epoch_top5_acc = compute_accuracy(data_loader, model, device=device, topk=(1, 5))
    print('top 1 acc: {:.3f}%'.format(epoch_top1_acc))
    print('top 5 acc: {:.3f}%'.format(epoch_top5_acc))


def test_top1_top5_err(data_loader, model, device):
    epoch_top1_err, epoch_top5_err = compute_accuracy(data_loader, model, device=device, isErr=True, topk=(1, 5))
    print('top 1 err: {:.3f}%'.format(epoch_top1_err))
    print('top 5 err: {:.3f}%'.format(epoch_top5_err))


if __name__ == '__main__':
    transform = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 提取测试集
    data_set = CIFAR10('./data', download=True, train=False, transform=transform)
    data_loader = DataLoader(data_set, shuffle=True, batch_size=128, num_workers=8)

    num_classes = 10
    model = alexnet(num_classes=num_classes)
    device = get_device()

    test_top1_top5_acc(data_loader, model, device)
    test_top1_top5_err(data_loader, model, device)
