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


def topk_accuracy(output, target, topk=(1,)):
    """
    计算前K个。N表示样本数，C表示类别数
    :param output: 大小为[N, C]，每行表示该样本计算得到的C个类别概率
    :param target: 大小为[N]，每行表示指定类别
    :param topk: tuple，计算前top-k的accuracy
    :return: list
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def compute_accuracy(data_loader, model, device=None, isErr=False):
    if device:
        model = model.to(device)

    epoch_top1_acc = 0.0
    epoch_top5_acc = 0.0
    for inputs, targets in data_loader:
        if device:
            inputs = inputs.to(device)
            targets = targets.to(device)

        # forward
        # track history if only in train
        with torch.no_grad():
            outputs = model(inputs)
            # print(outputs.shape)
            # _, preds = torch.max(outputs, 1)

            # statistics
            res_acc = topk_accuracy(outputs, targets, topk=(1, 5))
            epoch_top1_acc += res_acc[0]
            epoch_top5_acc += res_acc[1]

    if isErr:
        top_1_err = 1 - epoch_top1_acc / len(data_loader)
        top_5_err = 1 - epoch_top5_acc / len(data_loader)
        return top_1_err, top_5_err
    else:
        return epoch_top1_acc / len(data_loader), epoch_top5_acc / len(data_loader)


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
    epoch_top1_acc, epoch_top5_acc = compute_accuracy(data_loader, model, device=device)
    print('top 1 acc: {:.3f}'.format(epoch_top1_acc))
    print('top 5 acc: {:.3f}'.format(epoch_top5_acc))
