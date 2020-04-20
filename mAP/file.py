# -*- coding: utf-8 -*-

"""
@date: 2020/4/20 上午10:21
@file: file.py
@author: zj
@description: 
"""

import os
import glob
import json


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
