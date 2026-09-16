"""不下载权重的 RT-DETR 训练及多进程测试辅助函数。"""

import time

import torch

from ultralytics.nn.tasks import RTDETRDetectionModel


def tiny_rtdetr():
    """构造包含真实去噪、编码器和解码器分支的轻量模型。"""
    cfg = {
        "nc": 2,
        "backbone": [[-1, 1, "Conv", [16, 3, 2]], [-1, 1, "Conv", [32, 3, 2]]],
        "head": [[[1], 1, "RTDETRDecoder", [2, 32, 8, 4, 4, 2, 64]]],
    }
    model = RTDETRDetectionModel(cfg, verbose=False, summary=False)
    model.nc = 2
    model.model[-1].num_denoising = 4
    return model


def detection_batch(empty=False):
    """生成两张图片及可选的目标，保持所有测试离线。"""
    return {
        "img": torch.rand(2, 3, 32, 32),
        "batch_idx": torch.empty(0) if empty else torch.tensor([0.0, 1.0]),
        "cls": torch.empty(0, 1) if empty else torch.tensor([[0.0], [1.0]]),
        "bboxes": torch.empty(0, 4) if empty else torch.tensor([[0.5, 0.5, 0.3, 0.3], [0.6, 0.5, 0.2, 0.3]]),
    }


def run_gloo_workers(worker, tmp_path, *args):
    """限制子进程运行时间，失败或超时后回收所有测试进程。"""
    context = torch.multiprocessing.spawn(
        worker, args=(2, f"file://{tmp_path / 'rendezvous'}", *args), nprocs=2, join=False
    )
    deadline = time.monotonic() + 90
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                raise AssertionError("Gloo 回归测试超过 90 秒")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
