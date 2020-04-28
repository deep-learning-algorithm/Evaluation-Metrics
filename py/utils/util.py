# -*- coding: utf-8 -*-

"""
@date: 2020/4/20 上午10:17
@file: util.py
@author: zj
@description: 
"""

import numpy as np
import math
import sys
import torch

def get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def error(msg):
    print(msg)
    sys.exit(0)
