# -*- coding: utf-8 -*-

"""
@date: 2020/5/27 下午2:19
@file: misc.py
@author: zj
@description: 
"""

import numpy as np

voc_labels = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
              'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
              'train', 'tvmonitor']

if __name__ == '__main__':
    np.savetxt('labels.txt', voc_labels, fmt='%s', delimiter=' ')
