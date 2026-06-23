# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import copy
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import uniform_

from ultralytics.utils import LOGGER
from ultralytics.utils.checks import IS_ASCEND

__all__ = "inverse_sigmoid", "multi_scale_deformable_attn_pytorch"

MultiScaleDeformableAttnFunction = None
_MSDA_IMPORT_ATTEMPTED = False
_MSDA_FASTPATH_WARNING_EMITTED = False


def _get_msda_fastpath_function():
    """Lazy-load MMCV's Ascend MSDA op after device visibility has been configured."""
    global MultiScaleDeformableAttnFunction, _MSDA_IMPORT_ATTEMPTED
    if not _MSDA_IMPORT_ATTEMPTED:
        _MSDA_IMPORT_ATTEMPTED = True
        try:
            from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction as msda_function
        except Exception:
            msda_function = None
        MultiScaleDeformableAttnFunction = msda_function
    return MultiScaleDeformableAttnFunction


def _get_msda_fastpath_unavailable_reason(
    value: torch.Tensor, embed_dims: int, num_queries: int, num_points: int
) -> str | None:
    """Return why the Ascend MSDA fast path is unavailable, or None when supported."""
    if not IS_ASCEND or value.device.type != "npu":
        return None
    if value.dtype not in (torch.float16, torch.float32):
        return f"dtype={value.dtype} 不受支持"
    if not 32 <= embed_dims <= 256:
        return f"embed_dims={embed_dims} 超出 [32, 256]"
    if embed_dims % 8 != 0:
        return f"embed_dims={embed_dims} 不能被 8 整除"
    if num_queries < 32:
        return f"num_queries={num_queries} 小于 32"
    if not 4 <= num_points <= 8:
        return f"num_points={num_points} 超出 [4, 8]"
    return None


def _warn_msda_fastpath_unavailable(reason: str):
    """Warn once when Ascend MSDA fast path cannot be used."""
    global _MSDA_FASTPATH_WARNING_EMITTED
    if not _MSDA_FASTPATH_WARNING_EMITTED:
        LOGGER.warning(f"Ascend MSDA fastpath 不可用（{reason}），将回退到 PyTorch 实现；后续不再重复提示。")
        _MSDA_FASTPATH_WARNING_EMITTED = True


def _raise_if_ascend_bf16_grid_sample(value: torch.Tensor, sampling_locations: torch.Tensor):
    """Reject Ascend BF16 fallback before calling grid_sample."""
    if (
        IS_ASCEND
        and value.device.type == "npu"
        and (value.dtype == torch.bfloat16 or sampling_locations.dtype == torch.bfloat16)
    ):
        raise RuntimeError(
            "Ascend NPU 上 multi_scale_deformable_attn_pytorch 不支持 BF16："
            "F.grid_sample 在昇腾上不支持 BF16，只支持 FP16 和 FP32。"
        )


_MSDA_FASTPATH_CACHE_MAX = 16
_MSDA_FASTPATH_CACHE: dict[tuple[str, int, tuple[tuple[int, int], ...]], tuple[torch.Tensor, torch.Tensor]] = {}


def _get_msda_fastpath_tensors(
    value_spatial_shapes: torch.Tensor | list | tuple, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get (spatial_shapes, level_start_index) tensors for MMCV fast path with lightweight caching.

    Cache is used when input shapes are Python sequences (typical decoder path), avoiding repeated tensor construction.
    For tensor input, we keep behavior stateless to avoid content-dependent sync-heavy key generation.
    """
    if isinstance(value_spatial_shapes, torch.Tensor):
        if value_spatial_shapes.device == device and value_spatial_shapes.dtype == torch.int32:
            spatial_shapes = value_spatial_shapes
        else:
            spatial_shapes = value_spatial_shapes.to(device=device, dtype=torch.int32)
        level_start_index = torch.cat([spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]])
        return spatial_shapes, level_start_index

    shape_key = tuple((int(h), int(w)) for h, w in value_spatial_shapes)
    device_index = device.index if device.index is not None else -1
    cache_key = (device.type, device_index, shape_key)
    cached = _MSDA_FASTPATH_CACHE.get(cache_key)
    if cached is not None:
        return cached

    spatial_shapes = torch.tensor(shape_key, device=device, dtype=torch.int32)
    level_start_index = torch.cat([spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]])
    if len(_MSDA_FASTPATH_CACHE) >= _MSDA_FASTPATH_CACHE_MAX:
        _MSDA_FASTPATH_CACHE.pop(next(iter(_MSDA_FASTPATH_CACHE)))
    _MSDA_FASTPATH_CACHE[cache_key] = (spatial_shapes, level_start_index)
    return spatial_shapes, level_start_index


def _get_clones(module, n):
    """Create a list of cloned modules from the given module.

    Args:
        module (nn.Module): The module to be cloned.
        n (int): Number of clones to create.

    Returns:
        (nn.ModuleList): A ModuleList containing n clones of the input module.

    Examples:
        >>> import torch.nn as nn
        >>> layer = nn.Linear(10, 10)
        >>> clones = _get_clones(layer, 3)
        >>> len(clones)
        3
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def bias_init_with_prob(prior_prob=0.01):
    """Initialize conv/fc bias value according to a given probability value.

    This function calculates the bias initialization value based on a prior probability using the inverse sigmoid
    (logit)
    function. It's commonly used in object detection models to initialize classification layers with a specific positive
    prediction probability.

    Args:
        prior_prob (float, optional): Prior probability for bias initialization.

    Returns:
        (float): Bias initialization value calculated from the prior probability.

    Examples:
        >>> bias = bias_init_with_prob(0.01)
        >>> print(f"Bias initialization value: {bias:.4f}")
        Bias initialization value: -4.5951
    """
    return float(-np.log((1 - prior_prob) / prior_prob))  # return bias_init


def linear_init(module):
    """Initialize the weights and biases of a linear module.

    This function initializes the weights of a linear module using a uniform distribution within bounds calculated from
    the output dimension. If the module has a bias, it is also initialized.

    Args:
        module (nn.Module): Linear module to initialize.

    Examples:
        >>> import torch.nn as nn
        >>> linear = nn.Linear(10, 5)
        >>> linear_init(linear)
    """
    bound = 1 / math.sqrt(module.weight.shape[0])
    uniform_(module.weight, -bound, bound)
    if hasattr(module, "bias") and module.bias is not None:
        uniform_(module.bias, -bound, bound)


def inverse_sigmoid(x, eps=1e-5):
    """Calculate the inverse sigmoid function for a tensor.

    This function applies the inverse of the sigmoid function to a tensor, which is useful in various neural network
    operations, particularly in attention mechanisms and coordinate transformations.

    Args:
        x (torch.Tensor): Input tensor with values in range [0, 1].
        eps (float, optional): Small epsilon value to prevent numerical instability.

    Returns:
        (torch.Tensor): Tensor after applying the inverse sigmoid function.

    Examples:
        >>> x = torch.tensor([0.2, 0.5, 0.8])
        >>> inverse_sigmoid(x)
        tensor([-1.3863,  0.0000,  1.3863])
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def multi_scale_deformable_attn_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: list,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """Implement multi-scale deformable attention in PyTorch.

    Folds the (num_levels, num_points) axes into a single num_total_points axis so every traced tensor stays at rank <=
    5, the maximum rank supported by CoreML's MIL converter. Numerically equivalent to the rank-6 reference
    implementation on CUDA and CPU.

    Args:
        value (torch.Tensor): Value tensor with shape (bs, num_keys, num_heads, embed_dims).
        value_spatial_shapes (list): Per-level spatial shapes as [(H_0, W_0), ..., (H_{L-1}, W_{L-1})].
        sampling_locations (torch.Tensor): Sampling locations with shape (bs, num_queries, num_heads, num_levels *
            num_points, 2).
        attention_weights (torch.Tensor): Attention weights with shape (bs, num_queries, num_heads, num_levels *
            num_points).

    Returns:
        (torch.Tensor): Output tensor with shape (bs, num_queries, num_heads * embed_dims).

    References:
        https://github.com/IDEA-Research/detrex/blob/main/detrex/layers/multi_scale_deform_attn.py
    """
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, _, num_total_points, _ = sampling_locations.shape
    num_levels = len(value_spatial_shapes)
    if num_total_points % num_levels:
        raise ValueError(f"num_total_points={num_total_points} must be divisible by num_levels={num_levels}.")
    num_points = num_total_points // num_levels
    _raise_if_ascend_bf16_grid_sample(value, sampling_locations)

    # Ascend fast path via MMCV fused op, fallback to Ultralytics handwritten path when unavailable/unsupported.
    fastpath_unavailable_reason = _get_msda_fastpath_unavailable_reason(value, embed_dims, num_queries, num_points)
    use_ascend_fastpath = IS_ASCEND and value.device.type == "npu" and fastpath_unavailable_reason is None
    msda_function = _get_msda_fastpath_function() if use_ascend_fastpath else None
    if use_ascend_fastpath and msda_function is None:
        fastpath_unavailable_reason = "MMCV MultiScaleDeformableAttnFunction 导入失败"
    if msda_function is not None:
        spatial_shapes, level_start_index = _get_msda_fastpath_tensors(value_spatial_shapes, value.device)
        return msda_function.apply(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations.reshape(bs, num_queries, num_heads, num_levels, num_points, 2),
            attention_weights.reshape(bs, num_queries, num_heads, num_levels, num_points),
            64,
        )
    if IS_ASCEND and value.device.type == "npu" and fastpath_unavailable_reason is not None:
        _warn_msda_fastpath_unavailable(fastpath_unavailable_reason)

    # (bs, num_keys, num_heads, embed_dims) -> tuple of (bs*num_heads, embed_dims, H*W) per level
    value_list = value.permute(0, 2, 3, 1).flatten(0, 1).split([h * w for h, w in value_spatial_shapes], dim=-1)
    # Map to grid_sample coords in [-1, 1] and split per level: tuple of (bs*num_heads, num_queries, num_points, 2)
    sampling_grids = (2 * sampling_locations - 1).permute(0, 2, 1, 3, 4).flatten(0, 1).split(num_points, dim=-2)

    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        value_l = value_list[level].reshape(bs * num_heads, embed_dims, h, w)
        sampling_value_list.append(
            F.grid_sample(value_l, sampling_grids[level], mode="bilinear", padding_mode="zeros", align_corners=False)
        )
    attention_weights = attention_weights.permute(0, 2, 1, 3).reshape(bs * num_heads, 1, num_queries, num_total_points)
    output = (
        (torch.cat(sampling_value_list, dim=-1) * attention_weights)
        .sum(-1)
        .view(bs, num_heads * embed_dims, num_queries)
    )
    return output.transpose(1, 2).contiguous()
