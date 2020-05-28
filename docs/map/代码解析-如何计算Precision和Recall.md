
# [代码解析]如何计算Precision和Recall

相关文件：`py/metrics/map/voc_map.py`

## 前提条件

* 每类的真值边界框个数
* `TP/FP`列表

## Precision/Recall

小于置信度阈值下的边界框将不参与计算（过滤掉了）。之前已对按置信度排序的边界框进行了`TP/FP`的判断，所以可以计算出指定置信度阈值下的`TP/FP`个数

比如置信度阈值为`0.5`，刚好等同于列表中第`9`个边界框的置信度，那么之后的边界框将被过滤，仅需统计前`10`个边界框对应的`TP/FP`个数。示例如下：

```
tp = np.sum(TP[:10])
fp = np.sum(FP[:10])
```

然后就可以计算对应阈值下的`Precision/Recall`了

```
Precision = tp / (tp + fp)
Recall = tp / num_ground_truth
```