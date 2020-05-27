#!/bin/bash

mkdir outputs

for split in trainval test; do
  #    echo ${split}
  python voc2coco.py \
    --ann_dir ../data/VOCdevkit/VOC2007/Annotations \
    --ann_ids ../data/VOCdevkit/VOC2007/ImageSets/Main/${split}.txt \
    --labels labels.txt \
    --output outputs/${split}.json \
    --ext xml
done
