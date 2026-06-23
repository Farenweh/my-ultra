
# where算子替换

“torch.where”是一种条件选择的API，在使用过程中会产生大量的随机访问内存。而“torch.lerp”是一种线性插值API，算子逻辑更为简单。通过将where中的二元条件选择（True/False）转换为lerp的连续权重（0/1），利用线性插值公式 \(1 - weight\) \* x + weight \* y 来模拟条件选择过程。即当condition本身为input与other的比较，可以根据condition去选择input或other时，可等价使用lerp替换。该替换的优化原理为，相比较where，lerp采用更简单的逻辑算子实现，能够极大减少反向算子的耗时。

例如：

```python
out = torch.where(x < y, y, x)  
```

替换方法如下：

```python
out = torch.lerp(x, y, (x < y).float())
```
