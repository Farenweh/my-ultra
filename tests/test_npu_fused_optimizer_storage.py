"""回归小通道 BN 在融合参数缓冲区中的存储布局与多步数值。"""

from copy import deepcopy

import pytest
import torch
from torch import nn

from ultralytics.utils.torch_utils import ModelEMA, prepare_npu_fused_optimizer


def test_npu_fused_storage_preparation_preserves_cpu_parameters():
    """CPU 模型不需要 NPU 格式转换，参数身份和值应保持不变。"""
    model = nn.Sequential(nn.BatchNorm1d(8), nn.BatchNorm2d(85), nn.BatchNorm2d(16, affine=False))
    before = [(id(parameter), parameter.detach().clone()) for parameter in model.parameters()]
    assert prepare_npu_fused_optimizer(model) == 0
    for parameter, (identity, value) in zip(model.parameters(), before):
        assert id(parameter) == identity
        torch.testing.assert_close(parameter, value, rtol=0, atol=0)


def test_ema_fallback_copies_nonpersistent_buffers(monkeypatch):
    """私有格式触发 foreach 回退时，DDP 非持久化 buffer 仍应与模型同步。"""
    model = nn.Linear(2, 2)
    model.register_buffer("_ultralytics_ddp_buffer_align_pad_0", torch.zeros(8), persistent=False)
    model.register_buffer("temporary_counter", torch.tensor(0), persistent=False)
    ema = ModelEMA(model)
    model._ultralytics_ddp_buffer_align_pad_0.fill_(3)
    model.temporary_counter.fill_(7)
    assert "_ultralytics_ddp_buffer_align_pad_0" not in model.state_dict()

    def unsupported(*args, **kwargs):
        raise RuntimeError("模拟私有格式不支持 foreach")

    monkeypatch.setattr(torch, "_foreach_lerp_", unsupported)
    ema.update(model)
    assert ema._foreach_update_disabled
    torch.testing.assert_close(ema.ema._ultralytics_ddp_buffer_align_pad_0, torch.full((8,), 3.0))
    assert int(ema.ema.temporary_counter) == 7
    model._ultralytics_ddp_buffer_align_pad_0.fill_(4)
    model.temporary_counter.fill_(8)
    ema.update(model)
    torch.testing.assert_close(ema.ema._ultralytics_ddp_buffer_align_pad_0, torch.full((8,), 4.0))
    assert int(ema.ema.temporary_counter) == 8


@pytest.mark.skipif(not hasattr(torch, "npu") or not torch.npu.is_available(), reason="需要可用的 Ascend NPU")
@pytest.mark.parametrize("channels", (8, 85))
def test_npu_fused_adamw_preserves_parameters_and_matches_multistep_resume(channels):
    """真实私有格式 BN 前向不得覆盖邻接参数，融合 AdamW 多步及恢复应与普通实现一致。"""
    import torch_npu

    torch.npu.set_device(0)
    old_internal_format = getattr(torch.npu.config, "allow_internal_format", False)
    old_jit_compile = not torch.npu.is_jit_compile_false()
    torch.npu.config.allow_internal_format = True
    torch.npu.set_compile_mode(jit_compile=False)

    def create_model():
        return nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, 2),
        ).to("npu:0").train()

    def groups(parameters):
        return [
            {
                "params": [parameter for name, parameter in parameters if parameter.ndim > 1],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    parameter for name, parameter in parameters if name.endswith("weight") and parameter.ndim == 1
                ]
            },
            {"params": [parameter for name, parameter in parameters if name.endswith("bias")]},
        ]

    def compare(actual_model, actual_optimizer, reference_parameters, reference_optimizer):
        for actual, reference in zip(actual_model.parameters(), reference_parameters):
            torch.testing.assert_close(actual.cpu(), reference.cpu(), rtol=1e-5, atol=1e-6)
            for state_name in ("exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(
                    actual_optimizer.state[actual][state_name].cpu(),
                    reference_optimizer.state[reference][state_name].cpu(),
                    rtol=1e-5,
                    atol=1e-6,
                )
            assert int(actual_optimizer.state[actual]["step"]) == int(reference_optimizer.state[reference]["step"])

    try:
        torch.manual_seed(20261018)
        model = create_model()
        identities = [id(parameter) for parameter in model.parameters()]
        initial_values = [parameter.detach().cpu().clone() for parameter in model.parameters()]
        assert prepare_npu_fused_optimizer(model) == 2
        assert prepare_npu_fused_optimizer(model) == 0
        assert [id(parameter) for parameter in model.parameters()] == identities
        for parameter, value in zip(model.parameters(), initial_values):
            torch.testing.assert_close(parameter.cpu(), value, rtol=0, atol=0)
        assert torch_npu.get_npu_format(model[1].weight) == 3
        assert torch_npu.get_storage_size(model[1].weight) == ((channels + 15) // 16) * 16

        named = list(model.named_parameters())
        reference_named = [(name, nn.Parameter(parameter.detach().cpu().to("npu:0"))) for name, parameter in named]
        reference_parameters = [parameter for _, parameter in reference_named]
        fused = torch_npu.optim.NpuFusedAdamW(groups(named), lr=0.001, weight_decay=0.0)
        reference = torch.optim.AdamW(groups(reference_named), lr=0.001, weight_decay=0.0, foreach=False)
        images = torch.randn(2, 3, 8, 8, device="npu:0")
        target = torch.tensor([[0.2, -0.4], [-0.3, 0.5]], device="npu:0")

        for step in range(3):
            fused.zero_grad()
            reference.zero_grad(set_to_none=True)
            before_forward = [parameter.detach().cpu().clone() for parameter in model.parameters()]
            with torch.autocast("npu", dtype=torch.float16):
                loss = (model(images).float() - target).square().mean() * 0.1
            for parameter, value in zip(model.parameters(), before_forward):
                torch.testing.assert_close(parameter.cpu(), value, rtol=0, atol=0)
            assert torch.isfinite(loss)
            loss.backward()
            for parameter, value, expected in zip(model.parameters(), before_forward, reference_parameters):
                torch.testing.assert_close(parameter.cpu(), value, rtol=0, atol=0)
                expected.grad = parameter.grad.detach().cpu().to("npu:0")
            fused.step()
            reference.step()
            compare(model, fused, reference_parameters, reference)
            if step == 1:
                snapshot = {
                    "model": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                    "optimizer": deepcopy(fused.state_dict()),
                }

        restored = create_model()
        restored.load_state_dict(snapshot["model"])
        prepare_npu_fused_optimizer(restored)
        resumed = torch_npu.optim.NpuFusedAdamW(groups(list(restored.named_parameters())), lr=0.001, weight_decay=0.0)
        resumed.load_state_dict(snapshot["optimizer"])
        for parameter, reference_parameter in zip(restored.parameters(), reference_parameters):
            parameter.grad = reference_parameter.grad.detach().cpu().to("npu:0")
        resumed.step()
        compare(restored, resumed, reference_parameters, reference)
    finally:
        torch.npu.config.allow_internal_format = old_internal_format
        torch.npu.set_compile_mode(jit_compile=old_jit_compile)
