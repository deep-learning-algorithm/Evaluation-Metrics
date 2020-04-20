# -*- coding: utf-8 -*-

"""
@date: 2020/4/20 上午10:21
@file: file.py
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
