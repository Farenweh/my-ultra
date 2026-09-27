"""DINOv3 非重叠 patch 卷积与线性投影的等价性回归。"""

import io
from contextlib import nullcontext

import pytest
import torch

from ultralytics.nn.modules.third_party.dinov3.dinov3.layers.patch_embed import PatchEmbed, project_nonoverlapping_patches
from ultralytics.utils.checks import IS_ASCEND


@pytest.mark.parametrize("height,width", [(32, 48), (32, 32), (35, 51)])
def test_patch_projection_output_and_gradients(height, width):
    """保持卷积权重形状，同时对齐原图、权重和偏置梯度。"""
    torch.manual_seed(11)
    conv = torch.nn.Conv2d(3, 32, 16, 16).double()
    image = torch.randn(2, 3, height, width, dtype=torch.float64, requires_grad=True)
    ref = conv(image).flatten(2).transpose(1, 2)
    actual = project_nonoverlapping_patches(image, conv)
    grad = torch.randn_like(actual)
    a_grad = torch.autograd.grad(actual, (image, conv.weight, conv.bias), grad, retain_graph=True)
    r_grad = torch.autograd.grad(ref, (image, conv.weight, conv.bias), grad)
    for a, r in zip((actual, *a_grad), (ref, *r_grad)):
        torch.testing.assert_close(a, r, rtol=1e-10, atol=1e-10)


def test_patch_embed_cpu_fallback_and_checkpoint():
    """旧 checkpoint 内的普通 Conv2d 自动升级，CPU 输出及参数键不变。"""
    from ultralytics.nn.modules.third_party.dinov3.dinov3.layers.patch_embed import _PatchProjection

    module = PatchEmbed(img_size=(32, 48), patch_size=16, embed_dim=32)
    module.proj = torch.nn.Conv2d(3, 32, 16, 16)
    image = torch.randn(2, 3, 33, 49)
    expected = module.proj(image).flatten(2).transpose(1, 2)
    torch.testing.assert_close(module(image), expected)
    assert set(module.state_dict()) == {"proj.weight", "proj.bias"}
    stream = io.BytesIO()
    torch.save(module, stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=False)
    assert isinstance(restored.proj, _PatchProjection)
    assert set(restored.state_dict()) == set(module.state_dict())
    torch.testing.assert_close(restored(image), expected)


@pytest.mark.skipif(not IS_ASCEND, reason="需要 Ascend NPU")
@pytest.mark.parametrize("dtype,batch,height,width", [
    (torch.float32, 1, 32, 48),
    (torch.float16, 2, 35, 51),
    (torch.bfloat16, 1, 35, 51),
])
def test_patch_embed_npu_always_linear_and_gradients(monkeypatch, dtype, batch, height, width):
    """小 batch、矩形、非整除输入、可训练权重与不同精度均走线性投影。"""
    import torch_npu  # 注册 NPU 后端。
    from ultralytics.nn.modules.third_party.dinov3.dinov3.layers import patch_embed

    torch_npu.npu.set_compile_mode(jit_compile=False)
    if dtype == torch.float32:
        # 参考卷积默认开启 HF32；关闭它以便按相同精度检查输出和梯度。
        monkeypatch.setattr(torch.npu.conv, "allow_hf32", False)
        monkeypatch.setattr(torch.npu.matmul, "allow_hf32", False)
    torch.manual_seed(23)
    module = PatchEmbed(img_size=(height, width), patch_size=16, embed_dim=32,
                        flatten_embedding=False).to("npu:0").train()
    calls = []

    def counted(image, projection):
        calls.append(image.shape[0])
        return project_nonoverlapping_patches(image, projection)

    monkeypatch.setattr(patch_embed, "project_nonoverlapping_patches", counted)
    image = torch.rand(batch, 3, height, width, device="npu:0", requires_grad=True)
    context = nullcontext() if dtype == torch.float32 else torch.autocast("npu", dtype=dtype)
    with context:
        actual = module(image)
        expected = torch.nn.functional.conv2d(image, module.proj.weight, module.proj.bias, stride=16).permute(0, 2, 3, 1)
    assert calls == [batch] and actual.dtype == dtype
    grad = torch.randn_like(actual)
    inputs = (image, module.proj.weight, module.proj.bias)
    actual_grads = torch.autograd.grad(actual, inputs, grad, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, inputs, grad)
    tolerance = {torch.float32: 1e-5, torch.float16: 2e-3, torch.bfloat16: 2e-2}[dtype]
    for a, r in zip((actual, *actual_grads), (expected, *expected_grads)):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a.float().cpu(), r.float().cpu(), rtol=tolerance, atol=tolerance)


@pytest.mark.skipif(not IS_ASCEND, reason="需要 Ascend NPU")
def test_patch_embed_npu_legacy_hooks_and_tracing(monkeypatch):
    """旧完整模型在冻结推理、hook 和 tracing 场景仍使用线性实现。"""
    from ultralytics.nn.modules.third_party.dinov3.dinov3.layers import patch_embed

    torch.npu.set_compile_mode(jit_compile=False)
    module = PatchEmbed(img_size=(32, 48), patch_size=(8, 16), embed_dim=32)
    module.proj = torch.nn.Conv2d(3, 32, (8, 16), (8, 16))
    stream = io.BytesIO()
    torch.save(module, stream)
    stream.seek(0)
    module = torch.load(stream, weights_only=False).npu().eval().requires_grad_(False)
    calls = []

    def counted(image, projection):
        calls.append(True)
        return project_nonoverlapping_patches(image, projection)

    monkeypatch.setattr(patch_embed, "project_nonoverlapping_patches", counted)
    image = torch.rand(1, 3, 35, 51, device="npu:0")
    pre_hook = module.proj.register_forward_pre_hook(lambda m, args: (args[0] * 2,))
    post_hook = module.proj.register_forward_hook(lambda m, args, result: result + 1)
    try:
        with torch.inference_mode(), torch.autocast("npu", dtype=torch.float16):
            actual = module(image)
            expected = (torch.nn.functional.conv2d(image * 2, module.proj.weight, module.proj.bias,
                                                  stride=(8, 16)) + 1).flatten(2).transpose(1, 2)
        assert calls == [True]
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=2e-3, atol=2e-3)
    finally:
        pre_hook.remove()
        post_hook.remove()
    with torch.no_grad():
        traced = torch.jit.trace(module, image)
        torch.testing.assert_close(traced(image).cpu(), module(image).cpu(), rtol=2e-4, atol=2e-4)
    graph = str(traced.inlined_graph)
    assert "aten::linear" in graph and "aten::_convolution" not in graph
