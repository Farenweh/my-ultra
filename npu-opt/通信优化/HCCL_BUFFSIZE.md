# HCCL\_BUFFSIZE

## 功能描述

此环境变量用于控制两个NPU之间共享数据的缓存区大小。单位为MB，取值需要大于等于1，默认值是200MB。

集合通信网络中，每一个HCCL通信域都会占用2\*HCCL\_BUFFSIZE大小的缓存区。若集群网络中存在较多的HCCL通信域，此缓存区占用量就会增多，可能存在影响模型数据正常存放的风险。此种场景下，可通过此环境变量减少通信域占用的缓存区大小。若业务的模型数据量较小，但通信数据量较大，则可通过此环境变量增大HCCL通信域占用的缓存区大小，提升数据通信效率。

大语言模型（LLM，Large Language Model）中的建议配置值如下，向上取整。

![](./figures/HCCL_BUFFSIZE_fig_01.png)

其中Micro Batch Size为每张卡上的batch size，Sequence Length为序列长度，Hidden Size为模型隐藏层的维度，Size of Data Type为当前模型数据类型所占的内存大小。

此环境变量一般用于以下场景：

- 动态shape网络场景。
- 开发人员调用集合通信库HCCL的C语言接口进行框架对接的场景，HCCL库的C语言接口可参见《HCCL集合通信库指南》中的“[API](https://www.hiascend.com/document/detail/zh/canncommercial/900/API/hcclug/hcclcpp_07_0001.html)”章节。

>[!NOTE]
>
>- 该环境变量申请的内存为HCCL独占，不可与其他业务内存复用。
>- 每个通信域占用“2\*HCCL\_BUFFSIZE”大小的内存，分别用于发送和接收数据。
>- 该资源按通信域粒度管理，每个通信域独占一组“2\*HCCL\_BUFFSIZE”大小的内存，保证多通信域并发算子互不影响。

## 配置示例

```shell
export HCCL_BUFFSIZE=200
```

## 使用场景

当业务部署内存不足时，根据功能描述中的配置策略进行配置，若配置小于策略值，则会导致通信性能下降。
