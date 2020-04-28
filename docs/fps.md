
# FPS

实际使用过程中，常用的一个评价标准就是`FPS(Frames Per Second)`，也就是每秒能够检测多少图像

## 实现原理

累加模型$epoch$次计算时间$total$，再计算`FPS`

$$
FPS = \frac {epoch}{total}
$$

## 实现

其实现文件：`py/fps.py`