# -*- coding: utf-8 -*-

"""
@date: 2020/5/25 下午2:54
@file: acc.py
@author: zj
@description: 
"""

import torch


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


@torch.no_grad()
def compute_accuracy(data_loader, model, device=None, isErr=False, topk=(1, 5)):
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
            res_acc = topk_accuracy(outputs, targets, topk=topk)
            epoch_top1_acc += res_acc[0]
            epoch_top5_acc += res_acc[1]

    if isErr:
        top_1_err = 100 - epoch_top1_acc / len(data_loader)
        top_5_err = 100 - epoch_top5_acc / len(data_loader)
        return top_1_err, top_5_err
    else:
        return epoch_top1_acc / len(data_loader), epoch_top5_acc / len(data_loader)
