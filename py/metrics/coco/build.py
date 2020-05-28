# -*- coding: utf-8 -*-

"""
@date: 2020/5/28 下午1:53
@file: build.py
@author: zj
@description: 
"""

import os
import glob
import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .coco_map import compute_gt
from .coco_map import compute_dt


def coco_evaluation(ground_truth_dir, detection_result_dir, output_dir):
    cocoGt = compute_gt(ground_truth_dir, output_dir)
    cocoDt = compute_dt(cocoGt, detection_result_dir, output_dir)

    # initialize CocoEval object
    E = COCOeval(cocoGt, cocoDt, iouType='bbox')
    # set parameters as desired
    # E.params.recThrs = ...
    # E.params.catIds = [1]
    # E.params.maxDets = [10, 100, 300]
    # E.params.iouThrs = [0.5]
    # run per image evaluation
    E.evaluate()
    # accumulate per image results
    E.accumulate()
    # display summary metrics of results
    E.summarize()

    # print(E.params.iouThrs)
    # print(E.params.recThrs)
    # print(E.stats)
