# -*- coding: utf-8 -*-

"""
@date: 2020/5/25 下午3:58
@file: build.py
@author: zj
@description: 
"""

from .voc_map import compute_precision_recall
from .voc_map import compute_tp_fp
from .voc_map import voc_ap2
from .misc import pretreat
from .misc import parse_ground_truth
from .misc import parse_detection_results


def voc_evaluation(ground_truth_dir, detection_result_dir, tmp_json_dir):
    pretreat(ground_truth_dir, detection_result_dir, tmp_json_dir)

    # 将.txt文件解析成json格式
    gt_per_classes_dict = parse_ground_truth(ground_truth_dir, tmp_json_dir)
    gt_classes = list(gt_per_classes_dict.keys())
    # let's sort the classes alphabetically
    gt_classes = sorted(gt_classes)
    n_classes = len(gt_classes)
    # print(gt_classes)
    # print(gt_per_classes_dict)

    dt_per_classes_dict = parse_detection_results(detection_result_dir, tmp_json_dir)

    MIN_OVERLAP = 0.5
    # 计算每个类别的tp/fp
    sum_AP = 0.0

    metrics = dict()
    for cate in gt_classes:
        tp, fp = compute_tp_fp(dt_per_classes_dict, tmp_json_dir, cate, MIN_OVERLAP=MIN_OVERLAP)

        prec, rec = compute_precision_recall(tp, fp, gt_per_classes_dict[cate])

        # ap, mrec, mprec = voc_ap(rec[:], prec[:])
        ap = voc_ap2(rec[:], prec[:])
        sum_AP += ap

        metrics[cate] = ap
        # class_name + " AP = {0:.2f}%".format(ap*100)
        # text = "{0:.2f}%".format(ap * 100) + " = " + cate + " AP "
        # print(text)
    mAP = sum_AP / n_classes
    metrics['map'] = mAP
    # text = "mAP = {0:.2f}%".format(mAP * 100)
    # print(text)

    return metrics
