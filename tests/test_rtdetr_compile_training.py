from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from tests.rtdetr_helpers import detection_batch, run_gloo_workers, tiny_rtdetr
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils.torch_utils import ModelEMA, unwrap_model


class _RecordingLoss(RTDETRDetectionLoss):
    """记录图外 loss 的去噪输入和数值，不持有反向计算图。"""

    def forward(self, *args, **kwargs):
        self.latest_meta = kwargs.get("dn_meta")
        losses = super().forward(*args, **kwargs)
        self.latest_losses = {key: value.detach().clone() for key, value in losses.items()}
        return losses


def _trainer(model, compiled=True):
    trainer = object.__new__(RTDETRTrainer)
    trainer.model = model
    trainer.args = SimpleNamespace(compile=compiled)
    return trainer


def _recording_model():
    model = tiny_rtdetr().train()
    model.criterion = _RecordingLoss(nc=2, use_vfl=True)
    return model


def test_rtdetr_compile_path_keeps_targets_and_denoising(monkeypatch):
    """即使编译器回退为普通模型，compile 分派也不能丢掉 GT 和去噪监督。"""
    model = _recording_model()
    calls = []
    prepare = model._prepare_targets

    def counted(batch):
        calls.append(batch)
        return prepare(batch)

    monkeypatch.setattr(model, "_prepare_targets", counted)
    trainer = _trainer(model)
    loss, _ = trainer._model_forward(detection_batch())
    loss.backward()
    assert len(calls) == 1
    assert model.criterion.latest_meta is not None
    assert model.criterion.latest_losses["loss_class_dn"] > 0
    assert model.model[-1].denoising_class_embed.weight.grad.abs().sum() > 0


def test_rtdetr_dynamo_matches_eager_losses_and_gradients():
    """真实 Dynamo 包装下，变化的 GT 数量及空目标都与 eager 等价。"""
    torch._dynamo.reset()
    original = _recording_model()
    reference = deepcopy(original)
    compiled = torch.compile(original, backend="eager")
    trainer = _trainer(compiled)
    for index, empty in enumerate((False, True, False)):
        batch = detection_batch(empty)
        if index == 2:
            batch["batch_idx"] = torch.tensor([0.0, 1.0, 1.0])
            batch["cls"] = torch.tensor([[0.0], [1.0], [0.0]])
            batch["bboxes"] = torch.cat([batch["bboxes"], torch.tensor([[0.3, 0.4, 0.1, 0.2]])])
        original.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        torch.manual_seed(32 + index)
        expected, expected_items = reference(batch)
        expected.backward()
        torch.manual_seed(32 + index)
        actual, actual_items = trainer._model_forward(batch)
        actual.backward()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_items, expected_items)
        torch.testing.assert_close(original.criterion.latest_losses, reference.criterion.latest_losses)
        assert (original.criterion.latest_meta is None) == empty
        for (name, parameter), (_, target) in zip(original.named_parameters(), reference.named_parameters()):
            assert parameter.grad is not None, name
            torch.testing.assert_close(parameter.grad, target.grad)
    torch._dynamo.reset()


def test_rtdetr_compiled_ema_checkpoint_and_training_reload(tmp_path):
    """顶层编译包装不进入 EMA 和 checkpoint，重载后可继续训练。"""
    model = tiny_rtdetr().train()
    compiled = torch.compile(model, backend="eager")
    trainer = _trainer(compiled)
    ema = ModelEMA(compiled)
    optimizer = torch.optim.SGD(compiled.parameters(), lr=1e-4)
    trainer._model_forward(detection_batch())[0].backward()
    optimizer.step()
    ema.update(compiled)
    plain = deepcopy(unwrap_model(compiled))
    plain.criterion = None
    assert all("_orig_mod" not in name for name in plain.state_dict())
    assert all("_orig_mod" not in name for name in ema.ema.state_dict())
    path = tmp_path / "compiled.pt"
    torch.save({"model": plain, "train_args": {}, "optimizer": optimizer.state_dict()}, path)
    loaded, checkpoint = load_checkpoint(path)
    # 正常 trainer 会重新设置数据集属性，直接模型测试保留其已知类别数。
    loaded.train()
    del loaded.criterion
    resumed_optimizer = torch.optim.SGD(loaded.parameters(), lr=1e-4)
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    loss = loaded(detection_batch())[0]
    loss.backward()
    resumed_optimizer.step()
    assert loss.isfinite()
    torch._dynamo.reset()


def _compiled_ddp_worker(rank, world_size, init_method):
    """编译后的动态 GT 分支必须仍通过 DDP 包装执行。"""
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=init_method, rank=rank, world_size=world_size, timeout=timedelta(seconds=60)
    )
    try:
        model = torch.compile(tiny_rtdetr(), backend="eager")
        trainer = _trainer(model)
        trainer.model = DistributedDataParallel(
            model, static_graph=trainer._get_ddp_static_graph(), find_unused_parameters=False
        )
        for empty in (False, True, rank == 0, False):
            trainer.model.zero_grad(set_to_none=True)
            loss = trainer._model_forward(detection_batch(empty))[0]
            loss.backward()
            assert loss.isfinite()
            assert all(p.grad is not None and p.grad.isfinite().all() for p in trainer.model.parameters())
    finally:
        dist.destroy_process_group()


def test_rtdetr_compiled_ddp_handles_changing_targets(tmp_path):
    """两 rank 编译训练在 GT 分支变化后可以继续反向。"""
    run_gloo_workers(_compiled_ddp_worker, tmp_path)


def test_rtdetr_disables_only_its_compiled_ddp_static_graph():
    """仅 RT-DETR 的动态去噪训练取消 static_graph。"""
    for compiled in (False, True):
        trainer = _trainer(None, compiled)
        assert trainer._get_ddp_static_graph() is False
        assert BaseTrainer._get_ddp_static_graph(trainer) is compiled


@pytest.mark.slow
def test_rtdetr_inductor_forward_backward():
    """使用真实 CPU Inductor 执行含去噪分支的前向和反向。"""
    torch._dynamo.reset()
    model = _recording_model()
    trainer = _trainer(torch.compile(model, backend="inductor"))
    loss, _ = trainer._model_forward(detection_batch())
    loss.backward()
    assert loss.isfinite()
    assert model.criterion.latest_meta is not None
    assert model.criterion.latest_losses["loss_class_dn"] > 0
    assert all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters())
    torch._dynamo.reset()
