
# 准确率和错误率

参考：

[eval.py](https://github.com/bearpaw/pytorch-classification/blob/cc9106d598ff1fe375cc030873ceacfea0499d77/utils/eval.py)

[ImageNet Example Accuracy Calculation](https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840)

[Top k error calculation](https://discuss.pytorch.org/t/top-k-error-calculation/48815)

## 定义

在分类任务中，最常用的评价指标是准确率（`Accuracy`）和错误率（`Error Rate`）

### 准确率

在分类任务中，计算准确率（`accuracy`）即是指计算`Top-1`正确率，也就是**分类概率最高的类别等于标记类别**的样本数除以样本总数，其计算公式如下：

$$
Acc = \frac {TP + TN}{TP+FP+TN+FN}
$$

### 错误率

错误率和准确率相对

* `Top-1`错误率指的是**分类概率最高的类别不等于标记类别**的样本数除以样本总数
* `Top-5`错误率指的是**标记类别不在分类概率最高的前5个类别中**的样本数除以样本总数

错误率的计算公式如下：

$$
Err = \frac {FP + FN} {TP+FP+TN+FN}
$$

所以推导如下：

$$
Err = 1 - Acc
$$

## 实现

### 关键函数

使用了几个关键`PyTorch`函数

1. [topk](https://pytorch.org/docs/stable/torch.html#torch.topk)
2. [mul](https://pytorch.org/docs/stable/torch.html#torch.mul)

#### topk

>torch.topk(input, k, dim=None, largest=True, sorted=True, out=None) -> (Tensor, LongTensor)

从输入张量`input`中按指定维度返回前`k`个最大值

* `input`：输入张量
* `k`：前`k`个最大元素值
* `dim`：指定维度
* `largest`：默认为`True`，返回最大值；否则返回最小值
* `sorted`：是否进行排序

其返回两个张量，第一个表示值，第二个表示下标

```
>>> import torch
>>> a = torch.randn(5)
>>> a
tensor([ 0.4377,  0.0466, -0.3709, -1.9199, -0.4040])
>>> 
>>> a.topk(3)
torch.return_types.topk(
values=tensor([ 0.4377,  0.0466, -0.3709]),
indices=tensor([0, 1, 2]))
>>> b, d = a.topk(3)
>>> b
tensor([ 0.4377,  0.0466, -0.3709])
>>> d
tensor([0, 1, 2])
```

#### mul

> torch.mul(input, other, out=None)

将输入张量`input`的每个元素与参数`other`相乘

$$
out_{i} = other \times input_{i}
$$

就是乘法操作

```
>>> d
tensor([0, 1, 2])
>>> 
>>> d*0.1
tensor([0.0000, 0.1000, 0.2000])
>>>
>>> d.mul(0.1)
tensor([0.0000, 0.1000, 0.2000])
```

### 使用

* 实现文件：`metrics/acc.py`
* 测试文件：`test_acc_err.py`
