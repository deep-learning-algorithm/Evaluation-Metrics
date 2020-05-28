
# [代码解析]如何计算TP和FP

相关文件：`py/metrics/map/voc_map.py`

## 前提条件

* 获取所有的预测边界框信息（`{"confidence": "0.999", "file_id": "cucumber_61", "bbox": [16, 42, 225, 163]}`）
* 按置信度从大到小排序：`value.sort(key=lambda x: float(x['confidence']), reverse=True)`

## TP/FP

下面计算每个预测边界框对应的是`TP`还是`FP`

* 首先创建列表`tp`和`fp`，其长度等同于边界框个数，默认为`0`
* 遍历每一个边界框（*在之前已按置信度从大到小排序*）
  * 计算该预测边界框与对应文件中的标注文件框的`IoU`
  * 判断其最大`IoU`是否超过`IoU`阈值（*默认为`0.5`*）
    * 若超过
      * 判断对应的真值边界框是否已被使用（在它之前是否存在符合`IoU>=Thresh_IoU`的情况）
        * 若未被使用，则设置为`TP, tp[idx]=1`
        * 若已被使用，则设置为`FP, fp[idx]=1`
    * 若未超过，则设置为`FP, fp[idx]=1`