# 融合优化器

一般而言，优化器会频繁进行h2d（host to device）的下发操作，以完成对模型参数的更新，频繁的下发会拖慢模型的训练。而这些操作的底层逻辑是相同的，只是作用的数据不同，因此，统一在Host侧完成操作，再统一下发到device上，能够显著减少h2d的操作，提高性能。不过，需要注意的是，因为融合后的统一下发会要求更大的内存空间，无法利用device上的小块内存，因此融合优化器可能会占用更多的内存。

当前昇腾AI处理器支持NPU亲和的融合优化器功能，针对表1中的优化器做了昇腾AI处理器亲和性优化，使性能提升。

**表 1**  融合优化器使用方法

|**优化器名称**|**样例原代码**|**修改后代码**|
|--|--|--|
|sgd|optimizer = **torch.optim.SGD**(model.parameters(), lr=args.lr)|optimizer = **torch_npu.optim.NpuFusedSGD**(model.parameters(), lr=args.lr)|
|adadelta|optimizer = **torch.optim.Adadelta**(model.parameters(), lr=args.lr)|optimizer = **torch_npu.optim.NpuFusedAdadelta**(model.parameters(), lr=args.lr)|
|lamb|optimizer = **torch.optim.Lamb**(model.parameters(), lr=args.lr)|optimizer = **torch_npu.optim.NpuFusedLamb**(model.parameters(), lr=args.lr)|
|adamw|optimizer = **torch.optim.AdamW**(model.parameters(), lr=args.lr)|optimizer = **torch_npu.optim.NpuFusedAdamW**(model.parameters(), lr=args.lr)|
|adamp|optimizer = **torch.optim.AdamP**(model.parameters(), lr=args.lr)|optimizer = **torch_npu.optim.NpuFusedAdamP**(model.parameters(), lr=args.lr)|
|bert_adam|optimizer = **BertAdam**(model.parameters(), lr=args.lr)|optimizer = **torch_npu.optim.NpuFusedBertAdam**(model.parameters(), lr=args.lr)|
|rmsprop|optimizer = **torch.optim.RMSprop**(model.parameters(), lr=args.lr)|optimizer = **torch_npu.optim.NpuFusedRMSprop**(model.parameters(), lr=args.lr)|
|rmsprop_tf|optimizer = **torch.optim.RMSpropTF**(model.parameters(), lr=args.lr)|optimizer = **torch_npu.optim.NpuFusedRMSpropTF**(model.parameters(), lr=args.lr)|

> **说明：** BertAdam通常来自pytorch\_pretrained\_bert.optimization等三方库，并非torch.optim原生接口。
