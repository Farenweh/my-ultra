# Nonzero算子替换

nonzero算子是深度学习框架中常用的索引类算子，核心功能是返回输入张量中非零元素的坐标索引，并按行优先顺序输出结果。是一个典型的访存密集型操作，对昇腾达芬奇架构不友好。因此，核心的替换思路就在于避免大量的访存操作。

例如：

```python
import torch
import torch_npu

shape = (1024, )
mask= torch.randint(-1, 2, shape).npu()
gt_inds = torch.randint(-1, 2, shape).npu()
tensor_a = torch.ones(shape).float().npu()
mask_inds = torch.nonzero(gt_inds > 0, as_tuple=False).squeeze(1)
tensor_sum = tensor_a[mask_inds].sum()
```

替换代码如下：

```python
import torch
import torch_npu

shape = (1024, )
mask = torch.randint(-1, 2, shape).npu()
gt_inds = torch.randint(-1, 2, shape).npu()
tensor_a = torch.ones(shape).float().npu()

# --- 优化点：完全消除 nonzero ---
# 直接生成 0/1 掩码（float类型），替代索引提取
# 这里的 (gt_inds > 0) 会生成 BoolTensor，.float() 将其转为 0.0 和 1.0
float_mask = (gt_inds > 0).float()

# 利用算术乘法实现求和
# tensor_a[mask_inds].sum() 在数学上完全等价于 (tensor_a * float_mask).sum()
tensor_sum = (tensor_a * float_mask).sum()
```
