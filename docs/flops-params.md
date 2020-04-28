
# 参数数目和Flops计算

论文中常常使用**参数个数**和**Flops**来表示模型性能指标

## 参数个数

`PyTorch`计算模型参数数目

```
def num_model(model):
    return sum(param.numel() for param in model.parameters())
```

在`PyTorch`实现中，通常使用`32`位浮点数作为数据类型，计算所有的参数数目$N$后，可进一步转换成模型大小（单位为`MB`），计算如下：

$$
Model_{size} = N * 4 / 1024 / 1024
$$

## Flops

`Flops(Floating Point Of Operations)`表示浮点运算次数

### GFlops

通常使用`GFlops`来衡量算法性能，其表示十亿（`=10^9`）次的浮点运算

### Flops vs Macs

参考：[What is the relationship between GMACs and GFLOPs? #16](https://github.com/sovrasov/flops-counter.pytorch/issues/16)

`Mac`表示一次乘加操作（`Multiply–accumulate operation`）,通常在硬件架构中使用其计算张量操作，所以`Mac = 2Flops -> Flops = 2Mac`

## 实现

[Lyken17/pytorch-OpCounter](https://github.com/Lyken17/pytorch-OpCounter)实现了Macs和参数数目的计算

### 安装

```
pip install --upgrade git+https://github.com/Lyken17/pytorch-OpCounter.git
```

### 使用

* `py/flops-params.py`