# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""LocateAnything固定形状Qwen MLP的NPU Graph快路径。"""

from __future__ import annotations

from typing import Any

import torch

from ultralytics.utils import LOGGER

_GRAPH_FAILURES: set[tuple[str, int]] = set()


class _MlpGraphModule(torch.nn.Module):
    """只包含纯Tensor计算的Qwen MLP，避免将DynamicCache和远程Python分支捕获入图。"""

    def __init__(self, mlp: torch.nn.Module) -> None:
        super().__init__()
        self.gate_proj = mlp.gate_proj
        self.up_proj = mlp.up_proj
        self.down_proj = mlp.down_proj
        self.act_fn = mlp.act_fn

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


def _mlp_graph_forward(self: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """对最常见的MTP q=12固定形状执行ACL Graph，其他形状保留eager。"""
    original = self.__class__._locate_graph_original_forward
    enabled = bool(getattr(self, "_locate_npu_graph_enabled", False))
    supported = (
        enabled
        and not self.training
        and not torch.is_grad_enabled()
        and hidden_states.device.type == "npu"
        and hidden_states.dtype in {torch.float16, torch.bfloat16}
        and hidden_states.ndim == 3
        and hidden_states.shape[0] >= 64
        and hidden_states.shape[1] == 12
    )
    if not supported:
        return original(self, hidden_states)

    failure_key = (hidden_states.device.type, hidden_states.device.index or 0)
    if failure_key in _GRAPH_FAILURES:
        return original(self, hidden_states)
    shape_key = (tuple(hidden_states.shape), hidden_states.dtype)
    compiled = getattr(self, "_locate_npu_graph_compiled", None)
    compiled_key = getattr(self, "_locate_npu_graph_shape", None)
    if compiled is not None and compiled_key != shape_key:
        return original(self, hidden_states)
    try:
        if compiled is None:
            module = _MlpGraphModule(self).eval()
            compiled = torch.compile(module, backend="npugraphs", dynamic=False, fullgraph=True)
            # 不通过nn.Module.__setattr__注册为子模块，保持state_dict/训练结构不变。
            object.__setattr__(self, "_locate_npu_graph_compiled", compiled)
            object.__setattr__(self, "_locate_npu_graph_shape", shape_key)
        return compiled(hidden_states)
    except Exception as error:
        _GRAPH_FAILURES.add(failure_key)
        object.__setattr__(self, "_locate_npu_graph_compiled", None)
        LOGGER.warning(f"LocateAnything Qwen MLP NPU Graph捕获失败，本进程回退eager：{type(error).__name__}: {error}")
        return original(self, hidden_states)


def configure_npu_graph(model: Any, enabled: bool) -> int:
    """幂等安装Qwen MLP Graph入口，返回启用的层数。"""
    count = 0
    modules = model.modules() if hasattr(model, "modules") else ()
    for module in modules:
        if module.__class__.__name__ != "Qwen2MLP":
            continue
        target_class = module.__class__
        if not enabled and not hasattr(target_class, "_locate_graph_original_forward"):
            continue
        if not hasattr(target_class, "_locate_graph_original_forward"):
            target_class._locate_graph_original_forward = target_class.forward
            target_class.forward = _mlp_graph_forward
        module._locate_npu_graph_enabled = bool(enabled)
        if not enabled and hasattr(module, "_locate_npu_graph_compiled"):
            object.__setattr__(module, "_locate_npu_graph_compiled", None)
            object.__setattr__(module, "_locate_npu_graph_shape", None)
        count += 1
    return count


def run_decode_graph(**kwargs: Any) -> None:
    """保留旧内部入口；decoder整图会遭遇DynamicCache和数据依赖分支，故不再捕获。"""
    return None


def clear_decode_graph_cache() -> None:
    """清理失败状态，主要供测试和模型切换使用。"""
    _GRAPH_FAILURES.clear()


__all__ = ("clear_decode_graph_cache", "configure_npu_graph", "run_decode_graph")
