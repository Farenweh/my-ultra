# 个人中文文档

本目录保存当前分支自定义模型的中文使用说明，以及 Ascend NPU 适配、基准和优化记录。它不属于
`docs/en`，因此不会进入 Ultralytics 官方 MkDocs 站点构建。

## 内容导航

- [自定义模型](./models/README.md)：模型概况、权重、Python/YAML 接口和训练、验证、推理示例。
- [NPU 优化资料](./npu-opt/优化记录/README.md)：Ascend 910B2 实测报告、结构化结果和复现脚本。
- [TorchNPU 自有 API](./npu-opt/TorchNPU自有API/torch_npu_list.md)：TorchNPU 算子接口资料。
- [NPU 框架特性](./npu-opt/框架特性/overview.md)：任务队列、内存、通信和编译相关资料。

所有命令默认从仓库根目录执行。模型文档以当前分支代码为准；NPU 报告记录具体测试时间点的环境和结果。
