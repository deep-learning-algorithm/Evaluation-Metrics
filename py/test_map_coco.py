# -*- coding: utf-8 -*-

"""
@date: 2020/5/27 下午4:26
@file: test_map_coco.py
@author: zj
@description: 
"""

from metrics.coco import coco_evaluation

if __name__ == '__main__':
    annotation_dir = '/home/zj/data/image-localization-dataset/training_images'
    detection_result_dir = './input/detection-results'
    output_dir = './input'
    coco_evaluation(annotation_dir, detection_result_dir, output_dir)
