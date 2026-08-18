# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""LocateAnything专用LoRA和FSDP2训练循环。"""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
import tempfile
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from ultralytics.cfg import get_save_dir
from ultralytics.engine.runtime import CallbackHost, initialize_distributed_runtime
from ultralytics.utils import LOGGER, callbacks
from ultralytics.utils.dist import build_torchrun_command, find_free_network_port, normalize_k8s_launch_config
from ultralytics.utils.torch_utils import get_torch_device_backend

from .compat import SUPPORTED_REVISION
from .data import LocateAnythingCollator, build_locate_dataset


@dataclass
class LocateTrainResult:
    """LocateAnything训练产物摘要。"""

    save_dir: str
    final_model: str
    last_checkpoint: str | None
    method: str
    steps: int
    epochs: int
    final_loss: float | None


class LocateAnythingTrainer(CallbackHost):
    """不依赖YOLO batch语义的原生PyTorch训练器。"""

    def __init__(
        self,
        *,
        model: Any,
        data: str | Path,
        method: str = "lora",
        device: str | int | None = None,
        epochs: int = 1,
        max_steps: int = -1,
        batch: int = 1,
        workers: int = 4,
        max_seq_length: int = 4096,
        gradient_accumulation_steps: int = 1,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 0,
        max_grad_norm: float = 1.0,
        save_steps: int = 100,
        output_dir: str | Path | None = None,
        resume: bool | str | Path = False,
        seed: int = 0,
        lora_rank: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.05,
        vision_lora_rank: int = 0,
        negative_ratio: float = 1.0,
        max_negative_classes: int = 32,
        callbacks_: dict | None = None,
    ) -> None:
        method = method.lower()
        if method not in {"lora", "full"}:
            raise ValueError("method必须是'lora'或'full'")
        if epochs < 1 or batch < 1 or gradient_accumulation_steps < 1:
            raise ValueError("epochs、batch和gradient_accumulation_steps必须大于0")
        if max_seq_length < 128 or max_seq_length > 4096:
            raise ValueError("首版SDPA训练要求128 <= max_seq_length <= 4096")
        self.owner = model
        self.model = model.model
        self.processor = model.processor
        self.model_name = model.model_name
        self.revision = model.revision
        self.data_path = str(data)
        self.method = method
        self.device_spec = _normalize_device_spec(device, model.device)
        self.device_type = _device_type(self.device_spec, model.device.type)
        self.device_ids = _device_ids(self.device_spec)
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.rank = int(os.getenv("RANK", "0"))
        self.local_rank = int(os.getenv("LOCAL_RANK", "-1"))
        self.epochs = epochs
        self.max_steps = max_steps
        self.batch_size = batch
        self.workers = workers
        self.max_seq_length = max_seq_length
        self.accumulate = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.save_steps = save_steps
        self.resume = resume
        self.seed = seed
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.vision_lora_rank = vision_lora_rank
        self.negative_ratio = negative_ratio
        self.max_negative_classes = max_negative_classes
        self.args = SimpleNamespace(
            save_dir=str(output_dir) if output_dir else None,
            project="",
            name="train",
            task="locateanything",
            mode="train",
            exist_ok=False,
            data=self.data_path,
            method=self.method,
            device=self.device_spec,
            epochs=self.epochs,
            batch=self.batch_size,
            max_seq_length=self.max_seq_length,
            resume=self.resume,
        )
        self.save_dir = get_save_dir(self.args)
        self.args.save_dir = str(self.save_dir)
        self.setup_callbacks(callbacks_)
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.last_checkpoint: Path | None = None
        self._checkpoint_progress: tuple[int, int, int] | None = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.metrics: dict[str, float] = {}

    def train(self) -> LocateTrainResult:
        """启动单进程训练或自动创建torchrun子进程。"""
        requested_world = len(self.device_ids)
        if requested_world > 1 and self.device_type not in {"cuda", "npu"}:
            raise ValueError("LocateAnything多设备训练仅支持CUDA或Ascend NPU")
        if self.method == "full" and requested_world < 2 and self.local_rank == -1:
            raise ValueError("全参数SFT必须指定至少两个CUDA或NPU设备以使用FSDP2")
        if requested_world > 1 and self.local_rank == -1:
            return self._launch_distributed(requested_world)
        return self._train_worker()

    def _launch_distributed(self, world_size: int) -> LocateTrainResult:
        """生成最小启动脚本并通过torchrun执行多卡训练。"""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        dist_dir = self.save_dir / ".dist"
        dist_dir.mkdir(parents=True, exist_ok=True)
        config_path = dist_dir / "config.json"
        result_path = self.save_dir / "train_result.json"
        config = self._serializable_config()
        config["output_dir"] = str(self.save_dir)
        config["result_path"] = str(result_path)
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=f"{id(self)}.py",
            prefix="_locate_train_",
            dir=dist_dir,
            delete=False,
            encoding="utf-8",
        ) as file:
            file.write(
                "from ultralytics.models.locateanything.train import distributed_train_from_config\n"
                f"distributed_train_from_config({str(config_path)!r})\n"
            )
            launcher = Path(file.name)
        k8s = normalize_k8s_launch_config(world_size)
        command = build_torchrun_command(
            runner=launcher,
            nproc_per_node=world_size,
            master_port=k8s.master_port if k8s else find_free_network_port(),
            nnodes=k8s.nnodes if k8s else 1,
            node_rank=k8s.node_rank if k8s else 0,
            master_addr=k8s.master_addr if k8s else None,
        )
        LOGGER.info("LocateAnything分布式启动命令：" + " ".join(command))
        self.owner.model.to("cpu")
        _empty_accelerator_cache(self.device_type)
        try:
            subprocess.run(command, check=True)
        finally:
            launcher.unlink(missing_ok=True)
        if not result_path.is_file():
            raise RuntimeError(f"LocateAnything子进程未生成训练结果：{result_path}")
        return LocateTrainResult(**json.loads(result_path.read_text(encoding="utf-8")))

    def _train_worker(self) -> LocateTrainResult:
        """在当前rank执行专用训练循环。"""
        distributed = self.local_rank >= 0 and self.world_size > 1
        device = self.owner.device
        if distributed:
            physical_id = self.device_ids[self.local_rank]
            device, _, _ = initialize_distributed_runtime(
                device_type=self.device_type,
                device_spec=",".join(str(x) for x in self.device_ids),
                local_rank=self.local_rank,
                rank=self.rank,
                world_size=self.world_size,
                dist_module=dist,
                accelerator_resolver=get_torch_device_backend,
                is_ascend=self.device_type == "npu",
            )
            if device.index != physical_id:
                raise RuntimeError("分布式设备映射不一致")
            if self.owner.device != device:
                self.owner.device = device
                self.model.to(device)
        else:
            device = (
                torch.device(self.device_type, self.device_ids[0]) if self.device_type != "cpu" else torch.device("cpu")
            )
            if self.device_type in {"cuda", "npu", "xpu"}:
                get_torch_device_backend(device).set_device(device.index)
            if self.owner.device != device:
                self.owner.device = device
                self.model.to(device)
        self.device = device
        _seed_everything(self.seed + self.rank)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        if self.rank == 0:
            (self.save_dir / "args.json").write_text(
                json.dumps(self._serializable_config(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
        self.run_callbacks("on_pretrain_routine_start")

        dataset_kwargs = {
            "seed": self.seed,
            "negative_ratio": self.negative_ratio,
            "max_negative_classes": self.max_negative_classes,
        }
        dataset = build_locate_dataset(self.data_path, **dataset_kwargs)
        sampler = DistributedSampler(dataset, shuffle=True, seed=self.seed) if distributed else None
        collator = LocateAnythingCollator(self.processor, max_seq_length=self.max_seq_length, block_size=6)
        self.train_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=self.workers,
            collate_fn=collator,
            pin_memory=self.device.type == "cuda",
        )
        self._configure_model(distributed)
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not trainable:
            raise RuntimeError("LocateAnything没有可训练参数")
        self.optimizer = torch.optim.AdamW(trainable, lr=self.learning_rate, weight_decay=self.weight_decay)
        total_updates = max(math.ceil(len(self.train_loader) / self.accumulate) * self.epochs, 1)
        if self.max_steps > 0:
            total_updates = self.max_steps
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda step: _lr_factor(step, total_updates, self.warmup_steps),
        )
        if self.resume:
            self._load_checkpoint(self._resolve_resume_path())

        self.run_callbacks("on_pretrain_routine_end")
        self.run_callbacks("on_train_start")
        self.optimizer.zero_grad(set_to_none=True)
        final_loss = None
        accumulated_loss = 0.0
        stop = self.max_steps > 0 and self.global_step >= self.max_steps
        for epoch in range(self.epoch, self.epochs):
            if stop:
                break
            self.epoch = epoch
            if sampler is not None:
                sampler.set_epoch(epoch)
            self.run_callbacks("on_train_epoch_start")
            for batch_index, batch in enumerate(self.train_loader):
                if batch_index < self.batch_in_epoch:
                    continue
                self.batch_in_epoch = batch_index
                self.run_callbacks("on_train_batch_start")
                window_start = (batch_index // self.accumulate) * self.accumulate
                window_end = min(window_start + self.accumulate, len(self.train_loader))
                accumulation_divisor = window_end - window_start
                should_step = batch_index + 1 == window_end
                sync_context = _gradient_sync_context(self.model, should_step)
                batch = _move_batch(batch, self.device, self.owner.dtype)
                with sync_context:
                    with _autocast_context(self.device, self.owner.dtype):
                        outputs = self.model(**batch)
                        if not torch.isfinite(outputs.loss.detach()):
                            raise FloatingPointError(f"LocateAnything训练loss非有限值：{outputs.loss.detach()}")
                        loss = outputs.loss / accumulation_divisor
                    loss.backward()
                accumulated_loss += float(outputs.loss.detach().float().cpu()) / accumulation_divisor
                if should_step:
                    final_loss, accumulated_loss = accumulated_loss, 0.0
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(trainable, self.max_grad_norm)
                    self.run_callbacks("optimizer_step")
                    self.optimizer.step()
                    self.scheduler.step()
                    self.run_callbacks("on_before_zero_grad")
                    self.optimizer.zero_grad(set_to_none=True)
                    self.run_callbacks("on_params_update")
                    self.global_step += 1
                    self.batch_in_epoch = batch_index + 1
                    self.metrics = {"train/loss": final_loss, "lr": self.scheduler.get_last_lr()[0]}
                    self._log_step()
                    if self.save_steps > 0 and self.global_step % self.save_steps == 0:
                        self._save_checkpoint()
                    if self.max_steps > 0 and self.global_step >= self.max_steps:
                        stop = True
                self.run_callbacks("on_train_batch_end")
                if stop:
                    break
            self.run_callbacks("on_train_epoch_end")
            if self.batch_in_epoch >= len(self.train_loader):
                self.epoch = epoch + 1
                self.batch_in_epoch = 0
            self.run_callbacks("on_fit_epoch_end")
            if stop:
                break

        if self.global_step and self._checkpoint_progress != (self.global_step, self.epoch, self.batch_in_epoch):
            self._save_checkpoint()
        final_dir = self._export_final()
        result = LocateTrainResult(
            save_dir=str(self.save_dir),
            final_model=str(final_dir),
            last_checkpoint=str(self.last_checkpoint) if self.last_checkpoint else None,
            method=self.method,
            steps=self.global_step,
            epochs=min(self.epoch + int(self.batch_in_epoch > 0), self.epochs),
            final_loss=final_loss,
        )
        if self.rank == 0:
            (self.save_dir / "train_result.json").write_text(
                json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8"
            )
        self.run_callbacks("on_train_end")
        self.run_callbacks("teardown")
        if distributed:
            dist.barrier()
            dist.destroy_process_group()
        return result

    def _configure_model(self, distributed: bool) -> None:
        """冻结/注入LoRA并按需包装DDP或FSDP2。"""
        self.model.train()
        self.model.config.use_cache = False
        self.model.language_model.config.use_cache = False
        self.model.language_model.block_size = 6
        self.model.language_model.causal_attn = False
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable({"use_reentrant": False})
        _disable_remote_auxiliary_training_return(self.model.language_model)

        if self.method == "lora":
            for parameter in self.model.language_model.parameters():
                parameter.requires_grad = False
            for parameter in self.model.vision_model.parameters():
                parameter.requires_grad = False
            self.model.wrap_llm_lora(
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
            )
            if hasattr(self.model.language_model, "get_input_embeddings"):
                embeddings = self.model.language_model.get_input_embeddings()
                self._embedding_clone_hook = embeddings.register_forward_hook(_clone_embedding_output)
            _disable_remote_auxiliary_training_return(self.model.language_model)
            if self.vision_lora_rank:
                self.model.wrap_backbone_lora(
                    r=self.vision_lora_rank,
                    lora_alpha=2 * self.vision_lora_rank,
                    lora_dropout=self.lora_dropout,
                )
            for parameter in self.model.mlp1.parameters():
                parameter.requires_grad = True
            if distributed:
                self.model = DistributedDataParallel(
                    self.model,
                    device_ids=[self.device.index],
                    broadcast_buffers=False,
                    find_unused_parameters=False,
                )
            return

        if not distributed:
            raise RuntimeError("全参数SFT必须在torchrun多进程环境中使用FSDP2")
        for parameter in self.model.parameters():
            parameter.requires_grad = True
        self._apply_fsdp2()

    def _apply_fsdp2(self) -> None:
        """自底向上为MoonViT、Qwen和根模型应用FSDP2。"""
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import MixedPrecisionPolicy

        if self.device.type == "npu":
            from torch_npu.distributed.fsdp import fully_shard
        else:
            from torch.distributed.fsdp import fully_shard

        mesh = init_device_mesh(self.device.type, (self.world_size,), mesh_dim_names=("fsdp",))
        policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=torch.bfloat16,
        )
        for layer in self.model.language_model.model.layers:
            fully_shard(layer, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
        vision_blocks = getattr(self.model.vision_model.encoder, "blocks", None)
        if vision_blocks is None:
            vision_blocks = self.model.vision_model.encoder.layers
        for block in vision_blocks:
            fully_shard(block, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
        fully_shard(self.model, mesh=mesh, mp_policy=policy, reshard_after_forward=True)

    def _save_checkpoint(self) -> None:
        """保存可跨world size恢复的训练状态。"""
        checkpoint = self.save_dir / "checkpoints" / f"step-{self.global_step}"
        checkpoint.mkdir(parents=True, exist_ok=True)
        checkpoint_epoch, checkpoint_batch = self.epoch, self.batch_in_epoch
        if self.train_loader is not None and checkpoint_batch >= len(self.train_loader):
            checkpoint_epoch, checkpoint_batch = checkpoint_epoch + 1, 0
        if self.method == "full":
            import torch.distributed.checkpoint as dcp
            from torch.distributed.checkpoint.state_dict import get_state_dict

            model_state, optimizer_state = get_state_dict(self.model, self.optimizer)
            dcp.save({"model": model_state, "optimizer": optimizer_state}, checkpoint_id=str(checkpoint / "dcp"))
        else:
            raw_model = _unwrap_model(self.model)
            trainable_state = {
                name: parameter.detach().cpu()
                for name, parameter in raw_model.named_parameters()
                if parameter.requires_grad
            }
            if self.rank == 0:
                torch.save(trainable_state, checkpoint / "trainable.pt")
                torch.save(self.optimizer.state_dict(), checkpoint / "optimizer.pt")
        torch.save(_rng_state(self.device.type), checkpoint / f"rng-rank{self.rank}.pt")
        if self.rank == 0:
            torch.save(self.scheduler.state_dict(), checkpoint / "scheduler.pt")
            (checkpoint / "trainer.json").write_text(
                json.dumps(
                    {
                        "global_step": self.global_step,
                        "epoch": checkpoint_epoch,
                        "batch_in_epoch": checkpoint_batch,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            (self.save_dir / "last_checkpoint").write_text(str(checkpoint), encoding="utf-8")
        if dist.is_initialized():
            dist.barrier()
        self.last_checkpoint = checkpoint
        self._checkpoint_progress = (self.global_step, checkpoint_epoch, checkpoint_batch)
        self.run_callbacks("on_model_save")

    def _load_checkpoint(self, checkpoint: Path) -> None:
        """恢复模型、优化器、scheduler和随机状态。"""
        if self.method == "full":
            import torch.distributed.checkpoint as dcp
            from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

            model_state, optimizer_state = get_state_dict(self.model, self.optimizer)
            state = {"model": model_state, "optimizer": optimizer_state}
            dcp.load(state, checkpoint_id=str(checkpoint / "dcp"))
            set_state_dict(
                self.model,
                self.optimizer,
                model_state_dict=state["model"],
                optim_state_dict=state["optimizer"],
            )
        else:
            raw_model = _unwrap_model(self.model)
            raw_model.load_state_dict(torch.load(checkpoint / "trainable.pt", map_location="cpu"), strict=False)
            self.optimizer.load_state_dict(torch.load(checkpoint / "optimizer.pt", map_location="cpu"))
        self.scheduler.load_state_dict(torch.load(checkpoint / "scheduler.pt", map_location="cpu"))
        trainer_state = json.loads((checkpoint / "trainer.json").read_text(encoding="utf-8"))
        self.global_step = int(trainer_state["global_step"])
        self.epoch = int(trainer_state["epoch"])
        self.batch_in_epoch = int(trainer_state.get("batch_in_epoch", 0))
        rng_path = checkpoint / f"rng-rank{self.rank}.pt"
        if rng_path.is_file():
            _set_rng_state(torch.load(rng_path, map_location="cpu", weights_only=False))
        self.last_checkpoint = checkpoint
        self._checkpoint_progress = (self.global_step, self.epoch, self.batch_in_epoch)

    def _export_final(self) -> Path:
        """导出LoRA组合产物或完整Safetensors模型。"""
        final_dir = self.save_dir / "final"
        if self.method == "full":
            from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

            model_state = get_model_state_dict(
                self.model,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
            if self.rank == 0:
                self.model.save_pretrained(
                    final_dir,
                    state_dict=model_state,
                    safe_serialization=True,
                    max_shard_size="5GB",
                )
                self.processor.save_pretrained(final_dir)
                _write_manifest(final_dir, self, artifact="full")
        elif self.rank == 0:
            from safetensors.torch import save_file

            raw_model = _unwrap_model(self.model)
            final_dir.mkdir(parents=True, exist_ok=True)
            raw_model.language_model.save_pretrained(
                final_dir / "llm_adapter", safe_serialization=True, save_embedding_layers=False
            )
            if self.vision_lora_rank:
                raw_model.vision_model.save_pretrained(
                    final_dir / "vision_adapter", safe_serialization=True, save_embedding_layers=False
                )
            connector = {name: value.detach().cpu().contiguous() for name, value in raw_model.mlp1.state_dict().items()}
            save_file(connector, final_dir / "connector.safetensors")
            self.processor.save_pretrained(final_dir)
            _write_manifest(final_dir, self, artifact="lora")
        if dist.is_initialized():
            dist.barrier()
        return final_dir

    def _resolve_resume_path(self) -> Path:
        """解析显式checkpoint或当前run的last_checkpoint。"""
        if isinstance(self.resume, (str, Path)) and str(self.resume).lower() not in {"true", "false"}:
            path = Path(self.resume)
        else:
            marker = self.save_dir / "last_checkpoint"
            if not marker.is_file():
                raise FileNotFoundError(f"未找到恢复标记{marker}")
            path = Path(marker.read_text(encoding="utf-8").strip())
        if not path.is_dir():
            raise FileNotFoundError(f"LocateAnything checkpoint不存在：{path}")
        return path

    def _log_step(self) -> None:
        """由rank 0记录训练标量。"""
        if self.rank != 0:
            return
        record = {"step": self.global_step, "epoch": self.epoch, **self.metrics, "time": time.time()}
        with (self.save_dir / "results.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        LOGGER.info(
            f"LocateAnything epoch={self.epoch + 1}/{self.epochs} step={self.global_step} "
            f"loss={self.metrics['train/loss']:.5f} lr={self.metrics['lr']:.3g}"
        )

    def _serializable_config(self) -> dict[str, Any]:
        """返回torchrun子进程所需的JSON配置。"""
        return {
            "model_name": self.model_name,
            "revision": self.revision,
            "data": self.data_path,
            "method": self.method,
            "device_spec": self.device_spec,
            "device_type": self.device_type,
            "epochs": self.epochs,
            "max_steps": self.max_steps,
            "batch": self.batch_size,
            "workers": self.workers,
            "max_seq_length": self.max_seq_length,
            "gradient_accumulation_steps": self.accumulate,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "max_grad_norm": self.max_grad_norm,
            "save_steps": self.save_steps,
            "resume": str(self.resume),
            "seed": self.seed,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "vision_lora_rank": self.vision_lora_rank,
            "negative_ratio": self.negative_ratio,
            "max_negative_classes": self.max_negative_classes,
        }


def distributed_train_from_config(config_path: str | Path) -> None:
    """torchrun子进程入口。"""
    from .model import LocateAnything

    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    local_rank = int(os.environ["LOCAL_RANK"])
    device_spec = config.pop("device_spec")
    ids = _device_ids(device_spec)
    device_type = config.pop("device_type")
    result_path = Path(config.pop("result_path"))
    output_dir = config.pop("output_dir")
    model_name = config.pop("model_name")
    revision = config.pop("revision")
    resume = config.get("resume")
    config["resume"] = resume not in {"False", "false", ""} and resume
    owner = LocateAnything(model_name, revision=revision, device=f"{device_type}:{ids[local_rank]}")
    trainer = LocateAnythingTrainer(
        model=owner,
        output_dir=output_dir,
        device=device_spec,
        callbacks_=callbacks.get_default_callbacks(),
        **config,
    )
    result = trainer.train()
    if int(os.environ.get("RANK", "0")) == 0 and Path(result.save_dir) / "train_result.json" != result_path:
        result_path.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """将batch移动到当前rank设备。"""
    moved = {}
    for key, value in batch.items():
        value = value.to(device, non_blocking=device.type == "cuda")
        moved[key] = value.to(dtype) if key == "pixel_values" else value
    return moved


def _autocast_context(device: torch.device, dtype: torch.dtype):
    """仅在加速器半精度训练时启用与模型dtype一致的autocast。"""
    if device.type in {"cuda", "npu", "xpu"} and dtype in {torch.float16, torch.bfloat16}:
        return torch.amp.autocast(device.type, dtype=dtype)
    return nullcontext()


def _gradient_sync_context(model: nn.Module, should_step: bool):
    """在梯度累积微步关闭DDP/FSDP2同步。"""
    if should_step:
        if hasattr(model, "set_requires_gradient_sync"):
            model.set_requires_gradient_sync(True)
        return nullcontext()
    if hasattr(model, "no_sync"):
        return model.no_sync()
    if hasattr(model, "set_requires_gradient_sync"):
        model.set_requires_gradient_sync(False)
    return nullcontext()


def _lr_factor(step: int, total_steps: int, warmup_steps: int) -> float:
    """线性warmup后余弦衰减。"""
    if warmup_steps and step < warmup_steps:
        return max((step + 1) / warmup_steps, 1e-8)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return max(0.5 * (1 + math.cos(math.pi * min(max(progress, 0.0), 1.0))), 0.0)


def _normalize_device_spec(device: str | int | None, current: torch.device) -> str:
    """保留多设备列表并补齐设备类型。"""
    if device is None:
        return f"{current.type}:{current.index or 0}" if current.type not in {"cpu", "mps"} else current.type
    value = str(device).strip()
    if value in {"cpu", "mps"}:
        return value
    for device_type in ("cuda", "npu", "xpu"):
        if value == device_type:
            return f"{device_type}:0"
    if ":" in value:
        return value
    return f"{current.type}:{value}"


def _device_type(device_spec: str, fallback: str) -> str:
    """从规范设备字符串读取后端类型。"""
    prefix = device_spec.split(":", 1)[0]
    return prefix if prefix in {"cpu", "cuda", "npu", "xpu", "mps"} else fallback


def _device_ids(device_spec: str) -> list[int]:
    """从cuda/npu设备字符串解析物理设备ID。"""
    if device_spec in {"cpu", "mps"}:
        return [0]
    values = device_spec.split(":", 1)[-1]
    try:
        return [int(x.strip()) for x in values.split(",") if x.strip()]
    except ValueError as error:
        raise ValueError(f"非法LocateAnything设备列表{device_spec!r}") from error


def _unwrap_model(model: nn.Module) -> nn.Module:
    """移除DDP包装。"""
    return model.module if isinstance(model, DistributedDataParallel) else model


def _clone_embedding_output(
    _module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor
) -> torch.Tensor:
    """保留梯度链并避免PEFT叶子embedding在视觉token原地替换时报错。"""
    return output.clone()


def _disable_remote_auxiliary_training_return(language_model: nn.Module) -> None:
    """关闭上游Qwen无labels时会访问未初始化pos_loss_list的包装层分支。"""
    base_model = language_model.get_base_model() if hasattr(language_model, "get_base_model") else language_model
    base_model.training = False


def _seed_everything(seed: int) -> None:
    """设置Python、NumPy和Torch随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _rng_state(device_type: str) -> dict[str, Any]:
    """采集当前rank随机状态。"""
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if device_type == "cuda" and torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if device_type == "npu" and hasattr(torch, "npu") and torch.npu.is_available():
        state["npu"] = torch.npu.get_rng_state_all()
    return state


def _set_rng_state(state: dict[str, Any]) -> None:
    """恢复当前rank随机状态。"""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if "npu" in state and hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.set_rng_state_all(state["npu"])


def _empty_accelerator_cache(device_type: str) -> None:
    """释放父进程持有的加速器显存。"""
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device_type == "npu" and hasattr(torch, "npu"):
        torch.npu.empty_cache()


def _write_manifest(path: Path, trainer: LocateAnythingTrainer, *, artifact: str) -> None:
    """写入可由LocateAnything重新加载的产物清单。"""
    manifest = {
        "format_version": 1,
        "artifact": artifact,
        "base_model": trainer.model_name,
        "revision": trainer.revision or SUPPORTED_REVISION,
        "method": trainer.method,
        "max_seq_length": trainer.max_seq_length,
        "lora_rank": trainer.lora_rank if artifact == "lora" else 0,
        "vision_lora_rank": trainer.vision_lora_rank if artifact == "lora" else 0,
    }
    (path / "locateanything.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = "LocateAnythingTrainer", "LocateTrainResult", "distributed_train_from_config"
