# 亲和API替换

## torch\_npu.npu\_confusion\_transpose

npu\_confusion\_transpose可以代替transpose + view的操作。

- torch原生代码示例

    ```python
    import torch  
    data = torch.rand(64, 3, 64, 128).cuda() 
    batch, channel, height, width = data.shape
    result = data.transpose(1, 2).contiguous().view(batch, height, channel * width)
    ```

- torch\_npu代码示例

    ```python
    import torch 
    import torch_npu 
    from torch_npu.contrib import transfer_to_npu  
    data = torch.rand(64, 3, 64, 128).npu() 
    batch, channel, height, width = data.shape 
    result = torch_npu.npu_confusion_transpose(data, (0, 2, 1, 3), (batch, height, channel * width), transpose_first=True)
    ```

## optimizer.clip\_grad\_norm\_fused\_

在替换为NPU亲和梯度裁剪API之前，请确保代码中已使用NPU亲和优化器。

- torch原生代码示例

    ```python
    import torch 
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr) 
    torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10, norm_type=2)
    ```

- torch\_npu代码示例

    ```python
    import torch 
    import torch_npu 
    from torch_npu.contrib import transfer_to_npu  
    optimizer = torch_npu.optim.NpuFusedAdamW(model.parameters(), lr = lr) 
    optimizer.clip_grad_norm_fused_(max_norm=10, norm_type=2)
    ```
