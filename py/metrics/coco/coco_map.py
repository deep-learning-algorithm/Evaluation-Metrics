# -*- coding: utf-8 -*-

"""
@date: 2020/5/28 下午2:01
@file: coco_map.py
@author: zj
@description: 
"""

import json
import glob
import os
import xmltodict

from pycocotools.coco import COCO

from .misc import parse_annotation
from .misc import file_lines_to_list


def compute_gt(annotation_dir, output_dir):
    xml_list = glob.glob(os.path.join(annotation_dir, '*.xml'))

    images = list()
    annotations = list()
    categories = list()

    cat_id_list = list()
    ann_id = 0
    is_cat_exist = False
    for id, xml_path in enumerate(xml_list, 0):
        img_id = id + 1
        ann_id = ann_id + 1
        with open(xml_path, 'rb') as f:
            xml_dict = xmltodict.parse(f)
            filename, width, height, cat_name, xmin, ymin, xmax, ymax = parse_annotation(xml_dict['annotation'])

            if cat_name in cat_id_list:
                cat_id = cat_id_list.index(cat_name) + 1
                is_cat_exist = True
            else:
                cat_id_list.append(cat_name)
                cat_id = len(cat_id_list)
                is_cat_exist = False

            width = xmax - xmin
            height = ymax - ymin

            # id从1开始
            images.append({'id': img_id, 'width': width, 'height': height, 'file_name': filename})
            annotations.append({'id': ann_id, 'image_id': img_id, 'category_id': cat_id, 'area': width * height,
                                'bbox': [xmin, ymin, width, height], 'iscrowd': 0, 'segmentation': []})
            if not is_cat_exist:
                categories.append({'id': cat_id, 'name': cat_name, 'supercategory': ''})

    coco_json = {'images': images, 'annotations': annotations, 'categories': categories}
    gt_json_path = os.path.join(output_dir, 'gt.json')
    with open(os.path.join(output_dir, 'gt.json'), 'w') as f:
        json.dump(coco_json, f)
    cocoGt = COCO(gt_json_path)
    return cocoGt


def compute_dt(cocoGt, detection_result_dir, output_dir):
    assert isinstance(cocoGt, COCO)

    img_ids = cocoGt.getImgIds()
    imgs = cocoGt.loadImgs(img_ids)
    img_id_dict = {}
    for img in imgs:
        img_id = img['id']
        file_name = img['file_name'].split('\\')[-1].split('.')[0]
        img_id_dict[file_name] = img_id

    cat_ids = cocoGt.getCatIds()
    cats = cocoGt.loadCats(cat_ids)
    cat_id_dict = {}
    for cat in cats:
        cat_id = cat['id']
        cat_name = cat['name']
        cat_id_dict[cat_name] = cat_id

    file_list = os.listdir(detection_result_dir)
    coco_results = []
    for filename in file_list:
        file_path = os.path.join(detection_result_dir, filename)
        img_name = filename.split('.')[0]

        lines = file_lines_to_list(file_path)
        for line in lines:
            cate, confidence, xmin, ymin, xmax, ymax = line.split(' ')
            image_id = img_id_dict[img_name]
            cate_id = cat_id_dict[cate]
            coco_results.append(
                {
                    "image_id": image_id,
                    "category_id": cate_id,
                    # to xywh format
                    "bbox": [float(xmin), float(ymin), float(xmax) - float(xmin), float(ymax) - float(ymin)],
                    "score": float(confidence),
                }
            )

    json_result_file = os.path.join(output_dir, 'dt.json')
    with open(json_result_file, "w") as f:
        json.dump(coco_results, f)
    cocoDt = cocoGt.loadRes(json_result_file)
    return cocoDt
