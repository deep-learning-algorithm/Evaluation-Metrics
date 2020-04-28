# -*- coding: utf-8 -*-

"""
@date: 2020/4/28 上午10:32
@file: accuracy.py
@author: zj
@description: 计算Top-1 correct rate
"""

import torch
from torch.utils.data import DataLoader
from torchvision.models import alexnet
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from utils import util


def accuracy(data_loader, model, device=None):
    if device:
        model = model.to(device)

    running_corrects = 0
    for inputs, targets in data_loader:
        if device:
            inputs = inputs.to(device)
            targets = targets.to(device)

        # forward
        # track history if only in train
        with torch.no_grad():
            outputs = model(inputs)
            # print(outputs.shape)
            _, preds = torch.max(outputs, 1)

        # statistics
        running_corrects += torch.sum(preds == targets.data)

    epoch_acc = running_corrects.double() / len(data_loader.dataset)
    return epoch_acc


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

    device = util.get_device()
    acc = accuracy(data_loader, model, device=device)
    print('acc: {:.3f}'.format(acc))
