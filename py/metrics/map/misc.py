# -*- coding: utf-8 -*-

"""
@date: 2020/5/25 下午3:47
@file: misc.py
@author: zj
@description: 
"""

import numpy as np
import math
import sys
import torch
import os
import glob
import json
import shutil


def get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def error(msg):
    print(msg)
    sys.exit(0)


def pretreat(ground_truth_dir, detection_result_dir, tmp_json_dir):
    """
    预处理，保证真值边界框文件与预测边界框的文件一一对应，清空临时文件夹
    :param ground_truth_dir: 目录，保存真值边界框信息
    :param detection_result_dir: 目录，保存预测边界框信息
    :param tmp_json_dir: 临时文件夹
    """
    gt_list = [os.path.splitext(name)[0] for name in os.listdir(ground_truth_dir)]
    dr_list = [os.path.splitext(name)[0] for name in os.listdir(detection_result_dir)]

    if len(gt_list) == len(dr_list) and len(gt_list) == np.sum(
            [True if name in dr_list else False for name in gt_list]):
        pass
    else:
        error('真值边界框文件和预测边界框文件没有一一对应')

    if os.path.exists(tmp_json_dir):  # if it exist already
        # reset the tmp directory
        shutil.rmtree(tmp_json_dir)
    os.mkdir(tmp_json_dir)


def file_lines_to_list(path):
    """
     Convert the lines of a file to a list
    """
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content


def parse_ground_truth(ground_truth_dir, tmp_json_dir):
    """
    解析每个图片的真值边界框，以格式{"cate": "cucumber", "bbox": [23, 42, 206, 199], "used": true}保存
    """
    gt_path_list = glob.glob(os.path.join(ground_truth_dir, '*.txt'))

    # 统计每类的真值标注框数量
    gt_per_classes_dict = {}
    for gt_path in gt_path_list:
        json_list = list()
        lines = file_lines_to_list(gt_path)
        for line in lines:
            cate, xmin, ymin, xmax, ymax = line.split(' ')
            json_list.append({'cate': cate, 'bbox': [int(xmin), int(ymin), int(xmax), int(ymax)], 'used': False})

            if gt_per_classes_dict.get(cate) is None:
                gt_per_classes_dict[cate] = 1
            else:
                gt_per_classes_dict[cate] += 1
        # 保存
        name = os.path.splitext(os.path.basename(gt_path))[0]
        json_path = os.path.join(tmp_json_dir, name + ".json")
        with open(json_path, 'w') as f:
            json.dump(json_list, f)

    return gt_per_classes_dict


def parse_detection_results(detection_result_dir, tmp_json_dir):
    """
    解析每个类别的预测边界框，以格式{"confidence": "0.999", "file_id": "cucumber_61", "bbox": [16, 42, 225, 163]}保存
    """
    dr_path_list = glob.glob(os.path.join(detection_result_dir, '*.txt'))

    # 保存每个类别的预测边界框信息
    dt_per_classes_dict = dict()
    for dr_path in dr_path_list:
        lines = file_lines_to_list(dr_path)
        name = os.path.splitext(os.path.basename(dr_path))[0]

        for line in lines:
            cate, confidence, xmin, ymin, xmax, ymax = line.split(' ')
            if dt_per_classes_dict.get(cate) is None:
                dt_per_classes_dict[cate] = [
                    {'confidence': confidence, 'file_id': name, 'bbox': [int(xmin), int(ymin), int(xmax), int(ymax)]}]
            else:
                dt_per_classes_dict[cate].append(
                    {'confidence': confidence, 'file_id': name, 'bbox': [int(xmin), int(ymin), int(xmax), int(ymax)]})

    # 保存
    for key, value in dt_per_classes_dict.items():
        # 按置信度递减排序
        value.sort(key=lambda x: float(x['confidence']), reverse=True)

        json_path = os.path.join(tmp_json_dir, key + "_dt.json")
        with open(json_path, 'w') as f:
            json.dump(value, f)

    return dt_per_classes_dict
