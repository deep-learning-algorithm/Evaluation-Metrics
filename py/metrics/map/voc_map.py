# -*- coding: utf-8 -*-

"""
@date: 2020/4/20 下午3:30
@file: voc_map.py
@author: zj
@description: PASCAL VOC版本的mAP计算
"""

import json
import shutil
import os
import numpy as np


def compute_tp_fp(dt_per_classes_dict, tmp_json_dir, cate, MIN_OVERLAP=0.5):
    dt_list = dt_per_classes_dict[cate]
    # {cate_1: [], cate2: [], ...}
    nd = len(dt_list)
    tp = [0] * nd  # creates an array of zeros of size nd
    fp = [0] * nd
    # 遍历所有候选预测框，判断TP/FP/FN
    for idx, dt_data in enumerate(dt_list):
        # {"confidence": "0.999", "file_id": "cucumber_61", "bbox": [16, 42, 225, 163]}
        # 读取保存的信息
        file_id = dt_data['file_id']
        dt_bbox = dt_data['bbox']
        confidence = dt_data['confidence']

        # 读取对应文件的真值标注框
        gt_path = os.path.join(tmp_json_dir, file_id + ".json")
        gt_data = json.load(open(gt_path))

        # 逐个计算预测边界框和对应类别的真值标注框的IoU，得到其对应最大IoU的真值标注框
        ovmax = -1
        gt_match = -1
        # load detected object bounding-box
        for obj in gt_data:
            # {"cate": "cucumber", "bbox": [23, 42, 206, 199], "used": true}
            # 读取保存的信息
            obj_cate = obj['cate']
            obj_bbox = obj['bbox']
            obj_used = obj['used']

            # look for a class_name match
            if obj_cate == cate:
                bi = [max(dt_bbox[0], obj_bbox[0]), max(dt_bbox[1], obj_bbox[1]),
                      min(dt_bbox[2], obj_bbox[2]), min(dt_bbox[3], obj_bbox[3])]
                iw = bi[2] - bi[0] + 1
                ih = bi[3] - bi[1] + 1
                if iw > 0 and ih > 0:
                    # compute overlap (IoU) = area of intersection / area of union
                    ua = (dt_bbox[2] - dt_bbox[0] + 1) * (dt_bbox[3] - dt_bbox[1] + 1) + \
                         (obj_bbox[2] - obj_bbox[0] + 1) * (obj_bbox[3] - obj_bbox[1] + 1) \
                         - iw * ih
                    ov = iw * ih / ua
                    if ov > ovmax:
                        ovmax = ov
                        gt_match = obj
        # 如果大于最小IoU阈值，还需要进一步判断是否为TP
        if ovmax >= MIN_OVERLAP:
            if not bool(gt_match["used"]):
                # true positive
                tp[idx] = 1
                gt_match["used"] = True
                # update the ".json" file
                with open(gt_path, 'w') as f:
                    json.dump(gt_data, f)
            else:
                # false positive (multiple detection)
                fp[idx] = 1
        else:
            # false positive
            fp[idx] = 1

    return tp, fp


def compute_precision_recall(tp, fp, gt_per_classes_num):
    """
    计算不同阈值下的precision/recall
    """
    # compute precision/recall
    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val
    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val

    rec = tp[:]
    for idx, val in enumerate(tp):
        rec[idx] = float(val) / gt_per_classes_num
    # print(rec)
    prec = tp[:]
    for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
    return prec, rec


def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0)  # insert 0.0 at begining of list
    rec.append(1.0)  # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0)  # insert 0.0 at begining of list
    prec.append(0.0)  # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)  # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre


def voc_ap2(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
