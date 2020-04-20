
# mAP

目标检测任务下的`mAP`计算

## 操作步骤

1. 放置标注文件到`input/ground-truth`文件夹
2. 放置对应的预测结果到`input/detection-results`文件夹
3. 执行`python voc_map.py`

标注文件中每行表示一个标注边界框，其格式为

* `class-name xmin ymin xmax ymax`
  
预测文件中每行表示一个预测边界框，其格式为

* `class-name confidence xmin ymin xmax ymax`

## 相关阅读

* [[目标检测][PASCAL VOC]mAP](https://blog.zhujian.life/posts/d817618d.html)