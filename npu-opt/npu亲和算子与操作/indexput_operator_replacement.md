# IndexPut算子替换

由于昇腾SIMD（Single Instruction，Multiple Data，单指令，多数据）架构的特点，我们希望每个指令能够融合更多的数据，避免多次进行小数据的随机访问，因此，在遇到矩阵索引等操作时，可以对其进行替换，如替换成乘法操作。

例如：

```python
target_mask = (target < vocab_start_index) | (target >= vocab_end_index)  
masked_target = target.clone() - vocab_start_index  
masked_target[target_mask] = 0 
```

替换代码如下：

```python
target_mask = (target < vocab_start_index) | (target >= vocab_end_index)  
masked_target = target.clone() - vocab_start_index  
masked_target *= ~target_mask
```

将索引操作转换为矩阵的加乘操作，从而避免大量随机访存带来的性能劣化问题。
