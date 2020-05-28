# -*- coding: utf-8 -*-

"""
@date: 2020/5/27 下午9:41
@file: misc.py
@author: zj
@description: 
"""


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


def parse_annotation(ann_dict):
    filename = ann_dict['filename']

    size = ann_dict['size']
    width = size['width']
    height = size['height']

    object = ann_dict['object']
    cat_name = object['name']

    bndbox = object['bndbox']
    xmin = bndbox['xmin']
    ymin = bndbox['ymin']
    xmax = bndbox['xmax']
    ymax = bndbox['ymax']

    return filename, float(width), float(height), cat_name, float(xmin), float(ymin), float(xmax), float(ymax)
