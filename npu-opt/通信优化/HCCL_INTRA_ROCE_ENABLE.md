# HCCL\_INTRA\_ROCE\_ENABLE

## 功能描述

Atlas 200T A2 Box16 异构子框进行单server 16P业务部署，开启该环境变量后，两个8P之间使用RDMA链路代替SDMA链路作为mesh间互联链路。

## 配置示例

```shell
export HCCL_INTRA_ROCE_ENABLE=1
```

## 使用场景

当用户使用设备虚拟化场景或PCIe带宽较低时（如不足20GB/s），建议使用此参数。
