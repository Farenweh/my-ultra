from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from ultralytics.models.locateanything.npu_fast import (
    NpuFastPathState,
    _fused_qkv_projection,
    _fused_gate_up_projection,
    _qwen_mlp_forward,
    _vision_cumulative_lengths,
    configure_npu_kernel_fusions,
    install_npu_fast_path,
    normalize_npu_fast_policy,
    npu_fast_path_enabled,
)
from ultralytics.models.locateanything.npu_graph import configure_npu_graph


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, "auto"), (True, "auto"), (False, "off"), ("on", "auto"), ("strict", "strict")],
)
def test_normalize_npu_fast_policy(value, expected):
    assert normalize_npu_fast_policy(value) == expected


def test_normalize_npu_fast_policy_rejects_unknown_value():
    with pytest.raises(ValueError, match="npu_fast_path"):
        normalize_npu_fast_policy("always")


def test_cpu_auto_is_disabled_and_off_is_idempotent():
    model = torch.nn.Linear(2, 2)
    model._locate_npu_fast_policy = "auto"
    auto = install_npu_fast_path(model, "auto")
    assert auto == NpuFastPathState("auto", 0, 0, False, False)
    assert not npu_fast_path_enabled(model)
    first = install_npu_fast_path(model, "off")
    second = install_npu_fast_path(model, False)
    assert first == second == NpuFastPathState("off", 0, 0, False, False)
    assert model._locate_npu_fast_policy == "off"


def test_cpu_strict_requires_npu_model():
    model = SimpleNamespace(parameters=lambda: iter((torch.nn.Parameter(torch.ones(1)),)))
    with pytest.raises(RuntimeError, match="Ascend NPU"):
        install_npu_fast_path(model, "strict")


def test_fused_qkv_projection_matches_three_linears_and_refreshes_cache():
    module = SimpleNamespace(
        q_proj=torch.nn.Linear(4, 6),
        k_proj=torch.nn.Linear(4, 2),
        v_proj=torch.nn.Linear(4, 2),
    )
    values = torch.randn(3, 4)
    expected = tuple(projection(values) for projection in (module.q_proj, module.k_proj, module.v_proj))
    actual = _fused_qkv_projection(module, values)
    assert all(torch.allclose(left, right) for left, right in zip(actual, expected))
    first_cache = module._locate_qkv_cache
    with torch.no_grad():
        module.q_proj.weight.add_(1)
    refreshed = _fused_qkv_projection(module, values)
    assert module._locate_qkv_cache is not first_cache
    assert torch.allclose(refreshed[0], module.q_proj(values))


def test_disabling_kernel_fusions_releases_qkv_cache_without_changing_state_dict():
    class Qwen2SdpaAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(2, 2)
            self._locate_qkv_cache = ((), torch.ones(1), None)

    model = Qwen2SdpaAttention()
    keys = tuple(model.state_dict())
    configure_npu_kernel_fusions(model, fused_qkv=False, fused_add_rms_norm=False)
    assert model._locate_qkv_cache is None
    assert model._locate_npu_fused_qkv is False
    assert tuple(model.state_dict()) == keys


def test_fused_gate_up_projection_matches_two_linears_and_refreshes_cache():
    module = SimpleNamespace(
        gate_proj=torch.nn.Linear(4, 8, bias=False),
        up_proj=torch.nn.Linear(4, 8, bias=False),
    )
    values = torch.randn(2, 3, 4)
    expected = torch.cat((module.gate_proj(values), module.up_proj(values)), dim=-1)
    actual = _fused_gate_up_projection(module, values)
    assert torch.allclose(actual, expected)
    first_cache = module._locate_gate_up_cache
    with torch.no_grad():
        module.up_proj.weight.mul_(2)
    assert torch.allclose(_fused_gate_up_projection(module, values)[..., 8:], module.up_proj(values))
    assert module._locate_gate_up_cache is not first_cache


def test_vision_cumulative_lengths_are_bound_to_tensor_lifetime():
    first = torch.tensor([0, 2, 5], dtype=torch.int32)
    second = torch.tensor([0, 3, 7], dtype=torch.int32)
    assert _vision_cumulative_lengths(first) == (2, 5)
    assert _vision_cumulative_lengths(second) == (3, 7)
    assert first._locate_vision_cumulative_lengths == (2, 5)
    assert second._locate_vision_cumulative_lengths == (3, 7)


def test_npu_mlp_patch_keeps_training_forward_and_gradients():
    class Qwen2MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = torch.nn.Linear(4, 8, bias=False)
            self.up_proj = torch.nn.Linear(4, 8, bias=False)
            self.down_proj = torch.nn.Linear(8, 4, bias=False)
            self.act_fn = torch.nn.SiLU()

        def forward(self, values):
            return self.down_proj(self.act_fn(self.gate_proj(values)) * self.up_proj(values))

    Qwen2MLP._locate_original_forward = Qwen2MLP.forward
    Qwen2MLP.forward = _qwen_mlp_forward
    model = Qwen2MLP().train()
    model._locate_npu_fast_policy = "auto"
    model._locate_npu_fused_mlp = True
    keys = tuple(model.state_dict())
    values = torch.randn(2, 3, 4, requires_grad=True)
    model(values).sum().backward()
    assert values.grad is not None
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert tuple(model.state_dict()) == keys


def test_npu_graph_patch_keeps_training_state_dict_and_gradients():
    class Qwen2MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = torch.nn.Linear(4, 8, bias=False)
            self.up_proj = torch.nn.Linear(4, 8, bias=False)
            self.down_proj = torch.nn.Linear(8, 4, bias=False)
            self.act_fn = torch.nn.SiLU()

        def forward(self, values):
            return self.down_proj(self.act_fn(self.gate_proj(values)) * self.up_proj(values))

    model = Qwen2MLP()
    keys = tuple(model.state_dict())
    assert configure_npu_graph(model, True) == 1
    values = torch.randn(2, 3, 4, requires_grad=True)
    model.train()
    model(values).sum().backward()
    assert values.grad is not None
    assert tuple(model.state_dict()) == keys
    configure_npu_graph(model, False)
