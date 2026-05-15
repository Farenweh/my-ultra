from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import ultralytics.utils.torch_utils as torch_utils


class TinyHeadSplitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.ModuleList([nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)])

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x


class FakeCompiled(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self._orig_mod = module

    def forward(self, *args, **kwargs):
        return self._orig_mod(*args, **kwargs)


class TinyBNNet(nn.Module):
    def __init__(self, c: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(c, c)
        self.bn = nn.BatchNorm1d(c)
        self.fc2 = nn.Linear(c, c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.bn(self.fc1(x)))


_requires_attempt_compile_train = pytest.mark.skipif(
    not all(
        hasattr(torch_utils, name)
        for name in (
            "attempt_compile_train",
            "_is_ascend_train_prehead_compile_candidate",
            "_get_torchair_backend",
        )
    ),
    reason="train compile helpers are not implemented in torch_utils",
)
_requires_torchair_mode = pytest.mark.skipif(
    not hasattr(torch_utils, "_resolve_torchair_compile_mode"),
    reason="torchair compile mode helper is not implemented in torch_utils",
)
_requires_compile_state_helpers = pytest.mark.skipif(
    not all(
        hasattr(torch_utils, name)
        for name in (
            "canonical_state_dict",
            "make_plain_model",
            "has_compiled_children",
            "load_compile_state_dict",
        )
    ),
    reason="compile-state helpers are not implemented in torch_utils",
)


@_requires_attempt_compile_train
def test_attempt_compile_train_falls_back_to_attempt_compile(monkeypatch: pytest.MonkeyPatch):
    model = nn.Linear(4, 4)
    sentinel = nn.Linear(4, 4)
    calls = {}

    monkeypatch.setattr(torch_utils, "_is_ascend_train_prehead_compile_candidate", lambda *args, **kwargs: False)

    def _fake_attempt_compile(*args, **kwargs):
        calls["mode"] = kwargs["mode"]
        calls["device_type"] = kwargs["device"].type
        return sentinel

    monkeypatch.setattr(torch_utils, "attempt_compile", _fake_attempt_compile)
    out = torch_utils.attempt_compile_train(model, device=SimpleNamespace(type="cpu"), mode="reduce-overhead")

    assert out is sentinel
    assert calls == {"mode": "reduce-overhead", "device_type": "cpu"}


@_requires_attempt_compile_train
def test_attempt_compile_train_uses_ascend_prehead_torchair(monkeypatch: pytest.MonkeyPatch):
    model = TinyHeadSplitNet()
    calls = []

    monkeypatch.setattr(torch_utils, "_is_ascend_train_prehead_compile_candidate", lambda *args, **kwargs: True)
    monkeypatch.setattr(torch_utils, "_get_torchair_backend", lambda mode: ("torchair-backend", "max-autotune"))

    def _fake_compile(module, **kwargs):
        calls.append((type(module).__name__, kwargs.get("mode"), kwargs["backend"]))
        return FakeCompiled(module)

    monkeypatch.setattr(torch, "compile", _fake_compile)
    out = torch_utils.attempt_compile_train(model, device=SimpleNamespace(type="npu"), mode=True)

    assert out is model
    assert len(calls) == len(model.model) - 1
    assert calls[0][1:] == (None, "torchair-backend")
    assert all(hasattr(model.model[i], "_orig_mod") for i in range(len(model.model) - 1))
    assert not hasattr(model.model[-1], "_orig_mod")


@_requires_attempt_compile_train
def test_attempt_compile_train_ascend_failure_has_actionable_error(monkeypatch: pytest.MonkeyPatch):
    model = TinyHeadSplitNet()

    monkeypatch.setattr(torch_utils, "_is_ascend_train_prehead_compile_candidate", lambda *args, **kwargs: True)
    monkeypatch.setattr(torch_utils, "_get_torchair_backend", lambda mode: ("torchair-backend", "max-autotune"))

    def _fake_compile(module, **kwargs):
        if isinstance(module, nn.ReLU):
            raise RuntimeError("boom")
        return FakeCompiled(module)

    monkeypatch.setattr(torch, "compile", _fake_compile)

    with pytest.raises(RuntimeError, match=r"submodule 1 \(ReLU\).*compile=False"):
        torch_utils.attempt_compile_train(model, device=SimpleNamespace(type="npu"), mode="default")


@_requires_torchair_mode
def test_resolve_torchair_compile_mode_defaults_to_max_autotune():
    assert torch_utils._resolve_torchair_compile_mode(True) == (None, "max-autotune")
    assert torch_utils._resolve_torchair_compile_mode("default") == (None, "max-autotune")
    assert torch_utils._resolve_torchair_compile_mode("max-autotune") == (None, "max-autotune")
    assert torch_utils._resolve_torchair_compile_mode("reduce-overhead") == ("reduce-overhead", "reduce-overhead")


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile unavailable")
@_requires_compile_state_helpers
def test_compile_state_helpers_strip_child_wrapper_keys():
    model = TinyBNNet()
    model.fc1 = torch.compile(model.fc1, backend="eager")

    state = torch_utils.canonical_state_dict(model)
    plain = torch_utils.make_plain_model(model, copy=True)

    assert all("_orig_mod" not in key for key in state)
    assert all("_orig_mod" not in key for key in plain.state_dict())
    assert torch_utils.has_compiled_children(model) is True
    assert torch_utils.has_compiled_children(plain) is False


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile unavailable")
@_requires_compile_state_helpers
def test_load_compile_state_dict_round_trips_child_compiled_model():
    source = TinyBNNet()
    source.fc1 = torch.compile(source.fc1, backend="eager")

    target = TinyBNNet()
    target.fc1 = torch.compile(target.fc1, backend="eager")
    with torch.no_grad():
        for p in target.parameters():
            p.zero_()
        for b in target.buffers():
            if isinstance(b, torch.Tensor) and b.dtype.is_floating_point:
                b.zero_()

    src_state = torch_utils.canonical_state_dict(source)
    torch_utils.load_compile_state_dict(target, src_state)
    tgt_state = torch_utils.canonical_state_dict(target)

    assert src_state.keys() == tgt_state.keys()
    for key, value in src_state.items():
        if isinstance(value, torch.Tensor):
            assert torch.allclose(value, tgt_state[key])
