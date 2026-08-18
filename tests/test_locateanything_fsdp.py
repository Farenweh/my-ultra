from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch import nn

from ultralytics.utils.dist import build_torchrun_command, find_free_network_port


class _TinyTiedModel(nn.Module):
    """用于验证FSDP2和DCP的轻量共享权重替身。"""

    def __init__(self, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.embed = nn.Embedding(16, 8)
        self.layers = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])
        self.head = nn.Linear(8, 16, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.embed(tokens)
        for layer in self.layers:
            hidden = torch.relu(layer(hidden))
        return self.head(hidden)


def _fsdp2_checkpoint_worker(rank: int, world_size: int, port: int, checkpoint: str, mode: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    device_type = os.environ.get("LOCATE_FSDP_DEVICE", "cpu")
    if device_type == "npu":
        import torch_npu  # noqa: F401
    print(f"FSDP2替身 rank={rank} mode={mode} 初始化进程组", flush=True)
    if device_type == "npu":
        from ultralytics.engine.runtime import initialize_distributed_runtime
        from ultralytics.utils.torch_utils import get_torch_device_backend

        device, _, _ = initialize_distributed_runtime(
            device_type="npu",
            device_spec="npu:" + ",".join(str(index) for index in range(world_size)),
            local_rank=rank,
            rank=rank,
            world_size=world_size,
            dist_module=dist,
            accelerator_resolver=get_torch_device_backend,
            is_ascend=True,
        )
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        device = torch.device("cpu")
    from torch.distributed.checkpoint import load, save
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy

    if device_type == "npu":
        from torch_npu.distributed.fsdp import fully_shard
    else:
        from torch.distributed.fsdp import fully_shard

    model = _TinyTiedModel(seed=0 if mode == "save" else 99).to(device)
    mesh = init_device_mesh(device_type, (world_size,))
    policy = (
        MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, output_dtype=torch.bfloat16)
        if device_type == "npu"
        else None
    )
    print(f"FSDP2替身 rank={rank} mode={mode} 开始分片", flush=True)
    for layer in model.layers:
        fully_shard(layer, mesh=mesh, **({"mp_policy": policy} if policy else {}))
    fully_shard(model, mesh=mesh, **({"mp_policy": policy} if policy else {}))
    assert model.head.weight is model.embed.weight
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    checkpoint_path = Path(checkpoint)
    if rank == 0:
        (checkpoint_path / "dcp").mkdir(parents=True, exist_ok=True)
    dist.barrier()
    tokens = torch.tensor([[1, 2, 3], [3, 2, 1]], device=device)

    if mode == "save":
        print(f"FSDP2替身 rank={rank} 开始训练步", flush=True)
        loss = model(tokens).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model_state, optimizer_state = get_state_dict(model, optimizer)
        print(f"FSDP2替身 rank={rank} 开始保存", flush=True)
        save({"model": model_state, "optimizer": optimizer_state}, checkpoint_id=str(checkpoint_path / "dcp"))
        reference = model(tokens).detach().cpu()
        if rank == 0:
            torch.save(reference, checkpoint_path / "reference.pt")
    else:
        print(f"FSDP2替身 rank={rank} 开始恢复", flush=True)
        model_state, optimizer_state = get_state_dict(model, optimizer)
        state = {"model": model_state, "optimizer": optimizer_state}
        load(state, checkpoint_id=str(checkpoint_path / "dcp"))
        set_state_dict(model, optimizer, model_state_dict=state["model"], optim_state_dict=state["optimizer"])
        reference = torch.load(checkpoint_path / "reference.pt", weights_only=True)
        torch.testing.assert_close(model(tokens).detach().cpu(), reference)
        assert model.head.weight is model.embed.weight
    dist.barrier()
    print(f"FSDP2替身 rank={rank} mode={mode} 完成", flush=True)
    dist.destroy_process_group()


@pytest.mark.slow
def test_fsdp2_dcp_resume_across_world_size(tmp_path):
    """用两进程CPU FSDP2保存，再以单进程恢复，验证DCP重分片和共享权重。"""
    environment = {
        **os.environ,
        "LOCATE_FSDP_CHECKPOINT": str(tmp_path),
        "LOCATE_FSDP_MODE": "save",
    }
    subprocess.run(
        build_torchrun_command(runner=Path(__file__), nproc_per_node=2, master_port=find_free_network_port()),
        check=True,
        env=environment,
    )
    environment["LOCATE_FSDP_MODE"] = "load"
    subprocess.run(
        build_torchrun_command(runner=Path(__file__), nproc_per_node=1, master_port=find_free_network_port()),
        check=True,
        env=environment,
    )


if __name__ == "__main__":
    _fsdp2_checkpoint_worker(
        int(os.environ["RANK"]),
        int(os.environ["WORLD_SIZE"]),
        int(os.environ["MASTER_PORT"]),
        os.environ["LOCATE_FSDP_CHECKPOINT"],
        os.environ["LOCATE_FSDP_MODE"],
    )
