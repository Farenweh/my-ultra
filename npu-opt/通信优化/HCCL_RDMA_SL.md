# HCCL\_RDMA\_SL

## 功能描述

用于配置RDMA网卡的service level，该值需要和网卡配置的PFC优先级保持一致，若配置不一致可能导致性能劣化。

取值范围\[0,7\]，默认值4。

## 配置示例

```shell
# 优先级配置为3
export HCCL_RDMA_SL=3
```

## 使用场景

昇腾网卡与交换机QoS（Quality of Service，服务质量）不匹配导致RDMA通信带宽下降时配置。
