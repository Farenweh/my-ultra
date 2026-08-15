# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""受约束保护的TorchNPU自有算子辅助函数。"""

from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F

from ultralytics.utils.checks import IS_ASCEND

_torch_npu = None


def _get_torch_npu():
    """仅在真正选择NPU自有算子时导入torch_npu。"""
    global _torch_npu
    if _torch_npu is None:
        import torch_npu

        _torch_npu = torch_npu
    return _torch_npu


@lru_cache(maxsize=None)
def _npu_a2_device_supported(device_index: int) -> bool:
    """缓存设备是否属于npu_swiglu文档覆盖的Atlas A2产品。"""
    try:
        return "910B" in torch.npu.get_device_name(device_index).upper()
    except (AttributeError, RuntimeError):
        return False


def _supports_npu_swiglu(input: torch.Tensor, dim: int) -> bool:
    """检查输入是否满足npu_swiglu的公开约束。"""
    if not IS_ASCEND or input.device.type != "npu" or not 1 <= input.ndim <= 8 or not input.numel():
        return False
    if input.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        return False
    dim = dim if dim >= 0 else dim + input.ndim
    if not 0 <= dim < input.ndim or input.shape[dim] % 2 or not _npu_a2_device_supported(input.device.index or 0):
        return False
    try:
        return hasattr(_get_torch_npu(), "npu_swiglu")
    except (AttributeError, ImportError, RuntimeError):
        return False


def swiglu_with_npu_fallback(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """计算SwiGLU，并在Atlas A2输入满足约束时使用融合实现。"""
    if _supports_npu_swiglu(input, dim):
        return _get_torch_npu().npu_swiglu(input.contiguous(), dim=dim)
    first, second = input.chunk(2, dim=dim)
    return F.silu(first) * second


def _supports_npu_scatter_nd_update(input: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor) -> bool:
    """检查无梯度、唯一索引写入是否满足npu_scatter_nd_update_约束。"""
    if not IS_ASCEND or input.device.type != "npu" or not 1 <= input.ndim <= 8:
        return False
    if indices.ndim < 2 or indices.shape[-1] > input.ndim or indices.dtype not in {torch.int32, torch.int64}:
        return False
    if indices.device != input.device or updates.device != input.device or updates.dtype != input.dtype:
        return False
    supported_dtypes = {torch.float32, torch.float16, torch.bfloat16, torch.bool, torch.int64, torch.int8}
    if input.dtype not in supported_dtypes or input.requires_grad or updates.requires_grad:
        return False
    indexed_dims = indices.shape[-1]
    expected_shape = (*indices.shape[:-1], *input.shape[indexed_dims:])
    if updates.shape != expected_shape:
        return False
    try:
        return hasattr(_get_torch_npu(), "npu_scatter_nd_update_")
    except (AttributeError, ImportError, RuntimeError):
        return False


def scatter_nd_update_(input: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
    """按复合索引原位写入，支持时路由到Ascend ScatterNdUpdate。调用方必须保证索引唯一。"""
    if not indices.numel():
        return input
    if _supports_npu_scatter_nd_update(input, indices, updates):
        return _get_torch_npu().npu_scatter_nd_update_(input, indices, updates)
    input[tuple(indices.unbind(-1))] = updates
    return input


def _supports_npu_rms_norm(input: torch.Tensor, weight: torch.Tensor) -> bool:
    """检查RMSNorm是否满足npu_rms_norm约束且保持现有dtype语义。"""
    if not IS_ASCEND or input.device.type != "npu" or not 2 <= input.ndim <= 8:
        return False
    if input.dtype not in {torch.float16, torch.bfloat16, torch.float32} or weight.dtype != input.dtype:
        return False
    if weight.ndim < 1 or weight.ndim >= input.ndim or weight.device != input.device or not input.numel():
        return False
    if input.shape[-weight.ndim :] != weight.shape:
        return False
    try:
        return hasattr(_get_torch_npu(), "npu_rms_norm")
    except (AttributeError, ImportError, RuntimeError):
        return False


def rms_norm_with_npu_fallback(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """以现有FP32归一化语义计算RMSNorm，并在Ascend上使用融合实现。"""
    if _supports_npu_rms_norm(input, weight):
        return _get_torch_npu().npu_rms_norm(input, weight, epsilon=eps)[0]
    input_float = input.float()
    output = input_float * torch.rsqrt((input_float * input_float).mean(-1, keepdim=True) + eps)
    return output.type_as(input) * weight
