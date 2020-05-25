
# 如何衡量模型计算能力

## 定义

使用`Flops`可以衡量模型算法能力。`Flops(Floating Point Of Operations)`表示浮点运算次数，由于目前模型计算能力巨大，所以通常使用`GFlops`来衡量算法性能，其表示十亿（`=10^9`）次的浮点运算

### Flops vs Macs

参考：[What is the relationship between GMACs and GFLOPs? #16](https://github.com/sovrasov/flops-counter.pytorch/issues/16)

`Mac`表示一次乘加操作（`Multiply–accumulate operation`）,通常在硬件架构中使用其计算张量操作，所以`Mac = 2Flops -> Flops = 2Mac`

### 额外阅读

* [有关FLOPS的定义与计算](https://www.jianshu.com/p/e61eeae2d338)
* [分享一个FLOPs计算神器](https://www.jianshu.com/p/b1ceaa7effa8)：里面介绍了另外一个计算`Flops`的仓库

## 实现

使用了一个工具[Lyken17/pytorch-OpCounter](https://github.com/Lyken17/pytorch-OpCounter)完成`Macs`和参数数目的计算

### 安装

```
pip install --upgrade git+https://github.com/Lyken17/pytorch-OpCounter.git
```

### 使用

* 实现文件：`metrics/model.py`
* 测试文件：`test_params_flops.py`