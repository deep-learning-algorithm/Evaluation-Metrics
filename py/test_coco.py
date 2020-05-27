# -*- coding: utf-8 -*-

"""
@date: 2020/5/27 上午10:34
@file: test_coco.py
@author: zj
@description: 
"""

import os
import numpy as np
from matplotlib.pylab import plt
import skimage.io as io
from pycocotools.coco import COCO

VOC_ROOT = 'data/VOCdevkit/VOC2007'


def find_img_and_cat(coco):
    """
    根据随机的边界框信息找出对应的图像和类别信息
    :param coco:
    :return:
    """
    assert isinstance(coco, COCO)

    ann_ids = coco.getAnnIds()
    random_ann_id = ann_ids[np.random.randint(0, len(ann_ids))]
    ann = coco.loadAnns(ids=[random_ann_id])[0]

    img_ids = coco.getImgIds(imgIds=[ann['image_id']])
    img = coco.loadImgs(img_ids)[0]

    cat_ids = coco.getCatIds(catIds=[ann['category_id']])
    cat = coco.loadCats(cat_ids)[0]

    print(ann)
    print(img)
    print(cat)


def find_bbox_and_cat(coco):
    """
    根据随机的图像找出对应的边界框和类别信息
    :param coco:
    :return:
    """
    assert isinstance(coco, COCO)

    img_ids = coco.getImgIds()
    random_img_id = img_ids[np.random.randint(0, len(img_ids))]
    img = coco.loadImgs(ids=[random_img_id])

    ann_ids = coco.getAnnIds(imgIds=[random_img_id])
    anns = coco.loadAnns(ids=ann_ids)

    cat_ids = [ann['category_id'] for ann in anns]
    cats = coco.loadCats(ids=cat_ids)

    print(img)
    print(anns)
    print(cats)


def find_bbox_and_img(coco):
    """
    根据随机类别找出对应的边界框和图像信息
    :param coco:
    :return:
    """
    assert isinstance(coco, COCO)

    cat_ids = coco.getCatIds()
    random_cat_id = cat_ids[np.random.randint(0, len(cat_ids))]
    cat = coco.loadCats(ids=[random_cat_id])

    ann_ids = coco.getAnnIds(catIds=[random_cat_id])
    anns = coco.loadAnns(ids=ann_ids)

    img_ids = [ann['image_id'] for ann in anns]
    imgs = coco.loadImgs(ids=img_ids)

    print(cat)
    print(len(anns))
    print(len(imgs))


if __name__ == '__main__':
    annFile = 'voc2coco/outputs/test.json'
    coco = COCO(annFile)

    # find_img_and_cat(coco)
    # find_bbox_and_cat(coco)
    find_bbox_and_img(coco)
