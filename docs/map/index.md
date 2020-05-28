
# mAP

`mAP（mean Average Precision，即各类别AP的平均值）`是目标检测任务中最常用的评价标准

当前普遍使用两个数据集提供的`mAP`计算

1. `PASCAL VOC`
2. `COCO`

`PASCAL VOC`最开始系统化提出`mAP`的计算方式，而`COCO`的`mAP`计算更加复杂，也更加能够评估检测器性能

下面首先学习并实现`VOC mAP`的计算方式，然后学习`COCO`提供的工具包`cocoapi`

## VOC mAP

* 完整实现代码：`py/metrics/map/`
* 测试代码：`py/test_map_voc.py`

### 操作步骤

1. 放置标注文件到`input/ground-truth`文件夹
2. 放置对应的预测结果到`input/detection-results`文件夹
3. 执行`python voc_map.py`

标注文件中每行表示一个标注边界框，其格式为

* `class-name xmin ymin xmax ymax`
  
预测文件中每行表示一个预测边界框，其格式为

* `class-name confidence xmin ymin xmax ymax`

### 相关阅读

* [[目标检测][PASCAL VOC]mAP](https://blog.zhujian.life/posts/d817618d.html)

## COCO mAP

* [[数据集]COCO简介](https://blog.zhujian.life/posts/ef73a2c1.html)
* [ [数据集][COCO]目标检测任务](https://blog.zhujian.life/posts/46b5955b.html)
* [[数据集]voc2coco及cocoapi使用](https://blog.zhujian.life/posts/7bae9c2d.html)
* [[数据集][COCO]目标检测任务评估](https://blog.zhujian.life/posts/d77724ad.html)