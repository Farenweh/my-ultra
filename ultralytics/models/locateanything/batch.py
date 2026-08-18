# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""LocateAnything固定revision的SDPA批量hybrid生成引擎。

状态机、MTP窗口和逐行KV cache布局与NVIDIA LocateAnything固定revision公开的
``batch_utils``一致；本实现只保留PyTorch SDPA路径，不引入CUDA专用LA Flash或MagiAttention。
"""

from __future__ import annotations

import math
from copy import copy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import torch
from transformers.cache_utils import Cache

N_FUTURE = 6
SUPPORTED_SCHEDULERS = frozenset({"eager", "hold_ar", "ar_first", "pipeline", "adaptive"})
DECODE_BATCH_BUCKETS = (8, 16, 32, 64, 96, 128, 160, 192, 256)
_Q_SAMPLE_BUFFERS: dict[tuple[str, int, int], torch.Tensor] = {}
_SAMPLE_PARAMETER_BUFFERS: dict[tuple[str, int, int, torch.dtype, int, float], tuple[torch.Tensor, torch.Tensor]] = {}
_PAGED_CAUSAL_MASKS: dict[tuple[str, int], torch.Tensor] = {}
CANDIDATE_TOP_P_SIZE = 1024
CANDIDATE_TOP_P_RECHECK_STEPS = 64
VISUAL_TND_TOKEN_LIMIT = 16000


def _paged_right_down_causal_mask(device: torch.device) -> torch.Tensor:
    """按设备复用TorchNPU sparse_mode=3约定的2048因果mask。"""
    key = (device.type, device.index or 0)
    mask = _PAGED_CAUSAL_MASKS.get(key)
    if mask is None:
        mask = torch.triu(torch.ones((2048, 2048), dtype=torch.bool, device=device), diagonal=1)
        _PAGED_CAUSAL_MASKS[key] = mask
    return mask


def _sample_parameter_tensors(
    rows: int,
    top_k: int,
    top_p: float,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """复用采样算子的常量输入，避免每个decode step重复分配并填充小张量。"""
    key = (device.type, device.index or 0, int(rows), dtype, int(top_k), float(top_p))
    tensors = _SAMPLE_PARAMETER_BUFFERS.get(key)
    if tensors is None:
        tensors = (
            torch.full((rows,), top_k, dtype=torch.int32, device=device),
            torch.full((rows,), top_p, dtype=dtype, device=device),
        )
        _SAMPLE_PARAMETER_BUFFERS[key] = tensors
    return tensors


@dataclass
class BatchInput:
    """单个批量生成输入；图像预处理仍按图完成。"""

    input_ids: torch.Tensor
    pixel_values: torch.Tensor
    image_grid_hws: torch.Tensor | np.ndarray | None


@dataclass
class BatchOutput:
    """单个样本的生成文本与可精确统计的输出token数。"""

    text: str
    output_tokens: int
    forward_steps: int
    switch_to_ar: int
    stopped_repetition: bool = False


@dataclass
class PatternProbabilities:
    """仅保留hybrid MTP状态机需要的概率和top-k。"""

    box_start: torch.Tensor
    none: torch.Tensor
    box_end: torch.Tensor
    null_3: torch.Tensor
    null_4: torch.Tensor
    legal_frame: torch.Tensor
    coord_probs: torch.Tensor
    coord_ids: torch.Tensor
    ref_start: torch.Tensor
    ref_ids: torch.Tensor


class DeviceTokenHistory:
    """按活跃slot在设备上保存repetition penalty所需的完整token历史。"""

    def __init__(self, slots: int, capacity: int, fill_token_id: int, device: torch.device) -> None:
        self.slots = int(slots)
        self.capacity = max(int(capacity), 1)
        self.fill_token_id = int(fill_token_id)
        self.tokens = torch.full((self.slots, self.capacity), self.fill_token_id, dtype=torch.long, device=device)
        self.lengths = [0] * self.slots

    def _grow(self, required: int) -> None:
        if required <= self.capacity:
            return
        capacity = ((max(required, int(self.capacity * 1.5)) + 255) // 256) * 256
        tokens = self.tokens.new_full((self.slots, capacity), self.fill_token_id)
        tokens[:, : self.capacity].copy_(self.tokens)
        self.tokens = tokens
        self.capacity = capacity

    def reset_slots(self, slots: list[int], rows: list[list[int]]) -> None:
        """一次上传新补槽样本的prompt历史。"""
        if len(slots) != len(rows):
            raise ValueError("DeviceTokenHistory的slots与rows数量不一致")
        maximum = max((len(row) for row in rows), default=0)
        self._grow(maximum)
        if not slots:
            return
        host = torch.full((len(rows), maximum), self.fill_token_id, dtype=torch.long)
        for index, row in enumerate(rows):
            host[index, : len(row)] = torch.as_tensor(row, dtype=torch.long)
        slot_tensor = torch.as_tensor(slots, dtype=torch.long, device=self.tokens.device)
        self.tokens[slot_tensor, :maximum] = host.to(self.tokens.device)
        for slot, row in zip(slots, rows):
            self.lengths[slot] = len(row)

    def append(self, slots: list[int], rows: list[list[int]]) -> None:
        """整批增量写入本轮接受的token。"""
        if len(slots) != len(rows):
            raise ValueError("DeviceTokenHistory追加的slots与rows数量不一致")
        required = max((self.lengths[slot] + len(row) for slot, row in zip(slots, rows)), default=0)
        self._grow(required)
        flat_slots, flat_positions, flat_tokens = [], [], []
        for slot, row in zip(slots, rows):
            start = self.lengths[slot]
            flat_slots.extend([slot] * len(row))
            flat_positions.extend(range(start, start + len(row)))
            flat_tokens.extend(row)
            self.lengths[slot] += len(row)
        if flat_tokens:
            slot_tensor = torch.as_tensor(flat_slots, dtype=torch.long, device=self.tokens.device)
            position_tensor = torch.as_tensor(flat_positions, dtype=torch.long, device=self.tokens.device)
            values = torch.as_tensor(flat_tokens, dtype=torch.long, device=self.tokens.device)
            self.tokens[slot_tensor, position_tensor] = values

    def select(self, slots: list[int]) -> torch.Tensor:
        """返回当前组的设备端历史；右侧填充token不影响repetition集合语义。"""
        maximum = max((self.lengths[slot] for slot in slots), default=0)
        slot_tensor = torch.as_tensor(slots, dtype=torch.long, device=self.tokens.device)
        return self.tokens.index_select(0, slot_tensor)[:, :maximum]

    def release_slot(self, slot: int) -> None:
        self.lengths[int(slot)] = 0


class QSampleReservoir:
    """按活跃slot预生成多步qSample指数随机数，摊薄逐行随机算子启动开销。"""

    def __init__(self, slots: int, positions: int, device: torch.device) -> None:
        self.slots = int(slots)
        self.positions = max(int(positions), N_FUTURE)
        self.device = device
        self.values: torch.Tensor | None = None
        self.cursors = [self.positions] * self.slots

    def _ensure_values(self, vocabulary: int) -> torch.Tensor:
        if self.values is None or self.values.shape[-1] != vocabulary:
            self.values = torch.empty(
                (self.positions // N_FUTURE, self.slots, N_FUTURE, vocabulary),
                dtype=torch.float32,
                device=self.device,
            )
            self.cursors = [self.positions] * self.slots
        return self.values

    def take(
        self,
        future: int,
        slots: list[int],
        generators: list[torch.Generator],
        global_rows: list[int],
        vocabulary: int,
    ) -> torch.Tensor | None:
        """为连续slot的MTP返回零拷贝随机块；其他形状交回原始即时生成路径。"""
        if len(slots) != len(global_rows):
            raise ValueError("QSampleReservoir的slots与rows数量不一致")
        if future < 1 or future > self.positions:
            raise ValueError(f"qSample请求位置数必须位于[1,{self.positions}]，得到{future}")
        if future != N_FUTURE or not slots:
            return None
        start_slot = slots[0]
        if slots != list(range(start_slot, start_slot + len(slots))):
            return None
        cursor = self.cursors[start_slot]
        if any(self.cursors[slot] != cursor for slot in slots):
            return None
        values = self._ensure_values(vocabulary)
        for slot, global_row in zip(slots, global_rows):
            if not 0 <= slot < self.slots:
                raise ValueError(f"qSample slot越界：{slot}不在[0,{self.slots})")
            if self.cursors[slot] + future > self.positions:
                values[:, slot].exponential_(1.0, generator=generators[global_row])
                self.cursors[slot] = 0
        round_index = self.cursors[start_slot] // N_FUTURE
        selected = values[round_index, start_slot : start_slot + len(slots)]
        for slot in slots:
            self.cursors[slot] += future
        return selected.reshape(len(slots) * future, vocabulary)

    def release_slot(self, slot: int) -> None:
        """丢弃已结束样本未消费的随机数，避免新样本继承旧样本状态。"""
        self.cursors[int(slot)] = self.positions


class ExpandableStaticKVCache(Cache):
    """按活跃槽位持久保存KV，仅为当前forward拼接一次prefix与query。"""

    def __init__(self, slots: int, layers: int, initial_capacity: int) -> None:
        super().__init__(layers=[])
        self.slots = slots
        self.capacity = max(int(initial_capacity), 1)
        self.key_cache: list[torch.Tensor | None] = [None] * layers
        self.value_cache: list[torch.Tensor | None] = [None] * layers
        self.prefix_length = 0
        self.slot_indices: torch.Tensor | None = None
        self.write_positions: torch.Tensor | None = None
        self.write_mask: torch.Tensor | None = None
        self.prefix_lengths: list[int] = []

    def configure_step(
        self,
        slot_indices: torch.Tensor,
        prefix_length: int | list[int],
        write_positions: torch.Tensor,
        write_mask: torch.Tensor,
        required_capacity: int,
    ) -> None:
        """配置一次model forward的槽位、prefix与需要持久化的query token。"""
        self._grow(required_capacity)
        self.slot_indices = slot_indices
        if isinstance(prefix_length, list):
            self.prefix_lengths = [int(length) for length in prefix_length]
        else:
            self.prefix_lengths = [int(prefix_length)] * int(slot_indices.numel())
        self.prefix_length = max(self.prefix_lengths, default=0)
        self.write_positions = write_positions
        self.write_mask = write_mask.to(torch.bool)

    def _grow(self, required_capacity: int) -> None:
        if required_capacity <= self.capacity:
            return
        new_capacity = ((max(required_capacity, int(self.capacity * 1.5)) + 255) // 256) * 256
        for layer, (key, value) in enumerate(zip(self.key_cache, self.value_cache)):
            if key is None:
                continue
            key_new = key.new_empty((self.slots, key.shape[1], new_capacity, key.shape[3]))
            value_new = value.new_empty((self.slots, value.shape[1], new_capacity, value.shape[3]))
            key_new[:, :, : self.capacity].copy_(key)
            value_new[:, :, : self.capacity].copy_(value)
            self.key_cache[layer], self.value_cache[layer] = key_new, value_new
        self.capacity = new_capacity

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回attention所需prefix+query，并把真实query原位写回持久缓存。"""
        if self.slot_indices is None or self.write_positions is None or self.write_mask is None:
            raise RuntimeError("ExpandableStaticKVCache尚未配置当前step")
        if self.key_cache[layer_idx] is None:
            shape = (self.slots, key_states.shape[1], self.capacity, key_states.shape[3])
            self.key_cache[layer_idx] = key_states.new_empty(shape)
            self.value_cache[layer_idx] = value_states.new_empty(shape)
        key_cache = self.key_cache[layer_idx]
        value_cache = self.value_cache[layer_idx]
        if self.prefix_length:
            positions = torch.arange(self.prefix_length, device=key_states.device)[None, :]
            lengths = torch.as_tensor(self.prefix_lengths, dtype=torch.long, device=key_states.device)[:, None]
            visible = positions >= self.prefix_length - lengths
            logical_positions = (positions - (self.prefix_length - lengths)).clamp_min_(0)
            slots = self.slot_indices[:, None].expand_as(logical_positions)
            prefix_key = key_cache[slots, :, logical_positions, :].permute(0, 2, 1, 3).contiguous()
            prefix_value = value_cache[slots, :, logical_positions, :].permute(0, 2, 1, 3).contiguous()
            blocked = ~visible[:, None, :, None]
            prefix_key.masked_fill_(blocked, 0)
            prefix_value.masked_fill_(blocked, 0)
            attention_key = torch.cat((prefix_key, key_states), dim=2)
            attention_value = torch.cat((prefix_value, value_states), dim=2)
        else:
            attention_key, attention_value = key_states, value_states

        local_rows, query_positions = torch.where(self.write_mask)
        if local_rows.numel():
            slots = self.slot_indices.index_select(0, local_rows)
            positions = self.write_positions[local_rows, query_positions]
            key_cache[slots, :, positions, :] = key_states[local_rows, :, query_positions, :]
            value_cache[slots, :, positions, :] = value_states[local_rows, :, query_positions, :]
        return attention_key, attention_value

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.prefix_length

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        return self.capacity

    def get_max_length(self, layer_idx: int | None = None) -> int:
        return self.capacity

    def reset(self) -> None:
        self.prefix_length = 0


class PagedKVCache(Cache):
    """使用全局物理block池保存KV，在单token AR时直接交给NPU paged attention。"""

    def __init__(
        self,
        slots: int,
        layers: int,
        *,
        block_size: int,
        pool_blocks: int,
        max_seq_length: int,
        device: torch.device,
    ) -> None:
        super().__init__(layers=[])
        self.slots = int(slots)
        self.layers = int(layers)
        self.block_size = int(block_size)
        self.pool_blocks = max(int(pool_blocks), self.slots)
        self.max_blocks_per_sequence = max(math.ceil(max_seq_length / self.block_size), 1)
        self.key_cache: list[torch.Tensor | None] = [None] * self.layers
        self.value_cache: list[torch.Tensor | None] = [None] * self.layers
        self.block_table = torch.full((self.slots, self.max_blocks_per_sequence), -1, dtype=torch.int32, device=device)
        self._table_host = [[-1] * self.max_blocks_per_sequence for _ in range(self.slots)]
        self._allocated_blocks = [0] * self.slots
        self._free_blocks = list(range(self.pool_blocks - 1, -1, -1))
        self.slot_indices: torch.Tensor | None = None
        self.prefix_lengths: list[int] = []
        self.write_positions: torch.Tensor | None = None
        self.write_mask: torch.Tensor | None = None
        self.actual_seq_lengths: list[int] = []
        self.actual_query_lengths: list[int] = []
        self.active_block_table: torch.Tensor | None = None
        self.paged_attention_mask: torch.Tensor | None = None
        self.paged_sparse_mode = 0
        self._active_block_table_nd: torch.Tensor | None = None
        self._paged_attention_mask_nd: torch.Tensor | None = None
        self._sparse_causal_mask_nd: torch.Tensor | None = None
        self._npu_rope_cache: tuple[torch.Tensor, torch.Tensor] | None = None
        self._write_local_rows: torch.Tensor | None = None
        self._write_query_positions: torch.Tensor | None = None
        self._write_physical_blocks: torch.Tensor | None = None
        self._write_offsets: torch.Tensor | None = None
        self._scatter_indices: dict[int, torch.Tensor] = {}
        self.prefix_length = 0
        self.use_paged_attention = False
        self.direct_decode = True

    def configure_step(
        self,
        slot_indices: torch.Tensor,
        prefix_lengths: list[int],
        write_positions: torch.Tensor,
        write_mask: torch.Tensor,
        required_lengths: list[int],
        *,
        use_paged_attention: bool,
        slot_indices_host: list[int] | None = None,
    ) -> None:
        """分配当前step需要的block，并记录attention和写入元数据。"""
        slots = (
            [int(value) for value in slot_indices_host]
            if slot_indices_host is not None
            else [int(value) for value in slot_indices.detach().cpu().tolist()]
        )
        if not (len(slots) == len(prefix_lengths) == len(required_lengths) == write_positions.shape[0]):
            raise ValueError("PagedKVCache当前step的batch元数据大小不一致")
        table_rows, required_blocks = self._assign_blocks(slots, required_lengths, slot_indices)
        maximum_blocks = max(required_blocks, default=1)
        self.slot_indices = slot_indices
        self.prefix_lengths = [int(length) for length in prefix_lengths]
        self.write_positions = write_positions
        self.write_mask = write_mask.to(torch.bool)
        self.actual_seq_lengths = [int(length) for length in required_lengths]
        self.actual_query_lengths = [int(write_mask.shape[1])] * len(required_lengths)
        self.active_block_table = table_rows[:, :maximum_blocks].contiguous()
        write_valid = write_mask.to(torch.bool)
        if write_mask.shape[1] == 1:
            # 单token AR由IncreFlashAttention执行，不读取paged mask。
            self.paged_attention_mask = None
            self.paged_sparse_mode = 0
        elif write_mask.device.type == "npu":
            # MTP的真实q token始终是右对齐连续后缀，sparse_mode=3与逐batch显式mask等价。
            self.paged_attention_mask = _paged_right_down_causal_mask(write_mask.device)
            self.paged_sparse_mode = 3
        else:
            mask_width = maximum_blocks * self.block_size
            prefix_tensor = torch.as_tensor(self.prefix_lengths, dtype=torch.long, device=write_mask.device)[:, None]
            visible_counts = prefix_tensor + write_valid.to(torch.long).cumsum(dim=1)
            key_positions = torch.arange(mask_width, device=write_mask.device)[None, None, :]
            visible = (key_positions < visible_counts[:, :, None]) & write_valid[:, :, None]
            visible[:, :, 0] |= ~write_valid
            self.paged_attention_mask = (~visible)[:, None]
            self.paged_sparse_mode = 0
        # 这些张量在同一次decoder forward的所有层中完全相同，只计算一次。
        local_rows, query_positions = torch.where(self.write_mask)
        self._write_local_rows = local_rows
        self._write_query_positions = query_positions
        if local_rows.numel():
            write_slots = self.slot_indices.index_select(0, local_rows)
            write_positions_flat = self.write_positions[local_rows, query_positions]
            logical_blocks = torch.div(write_positions_flat, self.block_size, rounding_mode="floor")
            self._write_physical_blocks = self.block_table[write_slots, logical_blocks].long()
            self._write_offsets = write_positions_flat.remainder(self.block_size)
        else:
            self._write_physical_blocks = local_rows
            self._write_offsets = local_rows
        self._scatter_indices.clear()
        self._active_block_table_nd = None
        self._paged_attention_mask_nd = self._sparse_causal_mask_nd if self.paged_sparse_mode == 3 else None
        self._npu_rope_cache = None
        self.prefix_length = max(self.prefix_lengths, default=0)
        self.use_paged_attention = bool(use_paged_attention)

    def fork_current_step(self) -> PagedKVCache:
        """复制当前step元数据，同时共享物理KV池，供独立NPU stream安全排队prefill。"""
        forked = copy(self)
        forked._scatter_indices = {}
        return forked

    def _assign_blocks(
        self,
        slots: list[int],
        required_lengths: list[int],
        active_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[int]]:
        """仅在槽位跨越新block边界时分配并上传变化的device table行。"""
        required_blocks = [math.ceil(length / self.block_size) for length in required_lengths]
        maximum_required = max(required_blocks, default=1)
        if maximum_required > self.max_blocks_per_sequence:
            self._grow_block_table(maximum_required)
        missing = sum(max(blocks - self._allocated_blocks[slot], 0) for slot, blocks in zip(slots, required_blocks))
        if missing > len(self._free_blocks):
            self._grow_pool(missing - len(self._free_blocks))
        changed = []
        for slot, blocks in zip(slots, required_blocks):
            allocated = self._allocated_blocks[slot]
            if blocks <= allocated:
                continue
            for logical_block in range(allocated, blocks):
                self._table_host[slot][logical_block] = self._free_blocks.pop()
            self._allocated_blocks[slot] = blocks
            changed.append(slot)
        if changed:
            changed_rows = torch.as_tensor(
                [self._table_host[slot] for slot in changed], dtype=torch.int32, device=self.block_table.device
            )
            changed_indices = torch.as_tensor(changed, dtype=torch.long, device=self.block_table.device)
            self.block_table.index_copy_(0, changed_indices, changed_rows)
        if active_indices is None:
            active_indices = torch.as_tensor(slots, dtype=torch.long, device=self.block_table.device)
        return self.block_table.index_select(0, active_indices), required_blocks

    def _grow_block_table(self, required_blocks: int) -> None:
        """为后续到达的更长图像prompt扩展每个槽位的逻辑block表。"""
        new_maximum = ((int(required_blocks) + 15) // 16) * 16
        columns = new_maximum - self.max_blocks_per_sequence
        padding = self.block_table.new_full((self.slots, columns), -1)
        self.block_table = torch.cat((self.block_table, padding), dim=1)
        for row in self._table_host:
            row.extend([-1] * columns)
        self.max_blocks_per_sequence = new_maximum

    def _grow_pool(self, missing_blocks: int) -> None:
        """一次扩容多个block，避免长序列按slot预留8192-token空间。"""
        old_blocks = self.pool_blocks
        growth = max(int(old_blocks * 0.5), self.slots * 4, int(missing_blocks))
        new_blocks = old_blocks + growth
        for layer, (key, value) in enumerate(zip(self.key_cache, self.value_cache)):
            if key is None:
                continue
            shape = (new_blocks, key.shape[1], self.block_size, key.shape[3])
            new_key, new_value = key.new_empty(shape), value.new_empty(shape)
            new_key[:old_blocks].copy_(key)
            new_value[:old_blocks].copy_(value)
            self.key_cache[layer], self.value_cache[layer] = new_key, new_value
        self._free_blocks.extend(range(new_blocks - 1, old_blocks - 1, -1))
        self.pool_blocks = new_blocks

    def release_slot(self, slot: int) -> None:
        """回收已完成样本的物理block，供连续batch补槽复用。"""
        slot = int(slot)
        allocated = self._allocated_blocks[slot]
        self._free_blocks.extend(self._table_host[slot][:allocated])
        self._table_host[slot] = [-1] * self.max_blocks_per_sequence
        self._allocated_blocks[slot] = 0
        self.block_table[slot].fill_(-1)

    def import_row(self, slot: int, row_cache: tuple) -> int:
        """将一行动态KV一次迁移到paged池，用于唯一AR长尾。"""
        length = _row_kv_length(row_cache)
        self._assign_blocks([int(slot)], [length])
        blocks = math.ceil(length / self.block_size)
        for layer, (key, value) in enumerate(row_cache):
            if self.key_cache[layer] is None:
                shape = (self.pool_blocks, key.shape[1], self.block_size, key.shape[3])
                self.key_cache[layer] = key.new_empty(shape)
                self.value_cache[layer] = value.new_empty(shape)
            for logical_block in range(blocks):
                start = logical_block * self.block_size
                end = min(start + self.block_size, length)
                physical = self._table_host[int(slot)][logical_block]
                self.key_cache[layer][physical, :, : end - start].copy_(key[0, :, start:end])
                self.value_cache[layer][physical, :, : end - start].copy_(value[0, :, start:end])
        return length

    def export_row(self, slot: int, length: int) -> tuple:
        """将paged行还原为动态KV，仅在AR重新转回MTP时使用。"""
        blocks = math.ceil(length / self.block_size)
        result = []
        for key_pool, value_pool in zip(self.key_cache, self.value_cache):
            key_parts, value_parts = [], []
            for logical_block in range(blocks):
                start = logical_block * self.block_size
                take = min(self.block_size, length - start)
                physical = self._table_host[int(slot)][logical_block]
                key_parts.append(key_pool[physical : physical + 1, :, :take])
                value_parts.append(value_pool[physical : physical + 1, :, :take])
            result.append((torch.cat(key_parts, dim=2).contiguous(), torch.cat(value_parts, dim=2).contiguous()))
        return tuple(result)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """把真实q token写入block池；AR返回池，MTP/prefill返回等价左填充视图。"""
        if self.slot_indices is None or self.write_positions is None or self.write_mask is None:
            raise RuntimeError("PagedKVCache尚未配置当前step")
        if self.key_cache[layer_idx] is None:
            shape = (self.pool_blocks, key_states.shape[1], self.block_size, key_states.shape[3])
            self.key_cache[layer_idx] = key_states.new_empty(shape)
            self.value_cache[layer_idx] = value_states.new_empty(shape)
        key_pool, value_pool = self.key_cache[layer_idx], self.value_cache[layer_idx]

        local_rows = self._write_local_rows
        query_positions = self._write_query_positions
        if local_rows is None or query_positions is None:
            raise RuntimeError("PagedKVCache当前step缺少预计算写入索引")
        if local_rows.numel():
            physical_blocks = self._write_physical_blocks
            offsets = self._write_offsets
            if physical_blocks is None or offsets is None:
                raise RuntimeError("PagedKVCache当前step缺少预计算block索引")
            key_updates = key_states[local_rows, :, query_positions, :]
            value_updates = value_states[local_rows, :, query_positions, :]
            used_native_scatter = False
            if key_states.device.type == "npu":
                try:
                    import torch_npu

                    heads_count = key_states.shape[1]
                    indices = self._scatter_indices.get(heads_count)
                    if indices is None:
                        heads = torch.arange(heads_count, dtype=torch.long, device=key_states.device)
                        indices = torch.stack(
                            (
                                physical_blocks[:, None].expand(-1, heads_count),
                                heads[None, :].expand(len(physical_blocks), -1),
                                offsets[:, None].expand(-1, heads_count),
                            ),
                            dim=-1,
                        ).reshape(-1, 3)
                        self._scatter_indices[heads_count] = indices
                    torch_npu.npu_scatter_nd_update_(key_pool, indices, key_updates.reshape(-1, key_states.shape[-1]))
                    torch_npu.npu_scatter_nd_update_(
                        value_pool, indices, value_updates.reshape(-1, value_states.shape[-1])
                    )
                    used_native_scatter = True
                except (AttributeError, ImportError, RuntimeError):
                    used_native_scatter = False
            if not used_native_scatter:
                key_pool[physical_blocks, :, offsets, :] = key_updates
                value_pool[physical_blocks, :, offsets, :] = value_updates

        if self.use_paged_attention:
            return key_pool, value_pool
        if not self.prefix_length:
            return key_states, value_states
        positions = torch.arange(self.prefix_length, device=key_states.device)[None, :]
        lengths = torch.as_tensor(self.prefix_lengths, dtype=torch.long, device=key_states.device)[:, None]
        visible = positions >= self.prefix_length - lengths
        logical_positions = (positions - (self.prefix_length - lengths)).clamp_min_(0)
        logical_blocks = torch.div(logical_positions, self.block_size, rounding_mode="floor")
        offsets = logical_positions.remainder(self.block_size)
        table = self.block_table.index_select(0, self.slot_indices)
        physical_blocks = table.gather(1, logical_blocks).long()
        prefix_key = key_pool[physical_blocks, :, offsets, :].permute(0, 2, 1, 3).contiguous()
        prefix_value = value_pool[physical_blocks, :, offsets, :].permute(0, 2, 1, 3).contiguous()
        blocked = ~visible[:, None, :, None]
        prefix_key.masked_fill_(blocked, 0)
        prefix_value.masked_fill_(blocked, 0)
        return torch.cat((prefix_key, key_states), dim=2), torch.cat((prefix_value, value_states), dim=2)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.prefix_length

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        return self.max_blocks_per_sequence * self.block_size

    def get_max_length(self, layer_idx: int | None = None) -> int:
        return self.get_max_cache_shape()

    def reset(self) -> None:
        for slot in range(self.slots):
            self.release_slot(slot)
        self.prefix_length = 0
        self.use_paged_attention = False
        self.paged_attention_mask = None
        self.paged_sparse_mode = 0
        self._active_block_table_nd = None
        self._paged_attention_mask_nd = None
        self._npu_rope_cache = None
        self._write_local_rows = None
        self._write_query_positions = None
        self._write_physical_blocks = None
        self._write_offsets = None
        self._scatter_indices.clear()


def normalize_scheduler(value: str) -> str:
    """规范官方hybrid scheduler名称。"""
    aliases = {
        "default": "eager",
        "normal": "eager",
        "hold": "hold_ar",
        "hold-ar": "hold_ar",
        "repair_first": "ar_first",
        "repair-first": "ar_first",
        "ar-first": "ar_first",
    }
    scheduler = aliases.get(str(value).strip().lower(), str(value).strip().lower())
    if scheduler not in SUPPORTED_SCHEDULERS:
        choices = "、".join(sorted(SUPPORTED_SCHEDULERS))
        raise ValueError(f"scheduler必须是{choices}之一，得到{value!r}")
    return scheduler


def make_row_generators(device: torch.device, seeds: list[int]) -> list[torch.Generator]:
    """为每张图片创建独立加速器RNG，避免batch行顺序改变采样。"""
    generators = []
    for seed in seeds:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        generators.append(generator)
    return generators


def _row_kv_length(kv: tuple) -> int:
    return int(kv[0][0].shape[2])


def _ceil_bucket(value: int, buckets: tuple[int, ...], maximum: int) -> int:
    """返回不小于value且不超过maximum的最小固定桶。"""
    choices = sorted({bucket for bucket in buckets if 0 < bucket <= maximum} | {maximum})
    return next((bucket for bucket in choices if bucket >= value), maximum)


def pack_kv_rows(
    kv_rows: list[tuple | None],
    rows: list[int],
    device: torch.device,
    *,
    length_multiple: int = 1,
) -> tuple[tuple | None, torch.Tensor, list[int], int]:
    """左填充长度不同的逐行KV cache，以便在一次SDPA forward中运行。"""
    if isinstance(length_multiple, bool) or not isinstance(length_multiple, int) or length_multiple < 1:
        raise ValueError(f"length_multiple必须是正整数，得到{length_multiple!r}")
    lengths = [0 if kv_rows[row] is None else _row_kv_length(kv_rows[row]) for row in rows]
    max_length = max(lengths, default=0)
    if max_length:
        max_length = ((max_length + length_multiple - 1) // length_multiple) * length_multiple
    valid = torch.zeros((len(rows), max_length), dtype=torch.long, device=device)
    if max_length == 0:
        return None, valid, lengths, max_length

    reference = next(kv_rows[row] for row in rows if kv_rows[row] is not None)
    packed = []
    for layer in range(len(reference)):
        reference_key, _ = reference[layer]
        shape = (len(rows), reference_key.shape[1], max_length, reference_key.shape[3])
        packed_key = reference_key.new_zeros(shape)
        packed_value = reference_key.new_zeros(shape)
        for local_row, (row, length) in enumerate(zip(rows, lengths)):
            if length:
                key, value = kv_rows[row][layer]
                packed_key[local_row, :, -length:, :].copy_(key[0])
                packed_value[local_row, :, -length:, :].copy_(value[0])
        packed.append((packed_key, packed_value))
    length_tensor = torch.as_tensor(lengths, dtype=torch.long, device=device)
    valid = (torch.arange(max_length, device=device)[None, :] >= max_length - length_tensor[:, None]).long()
    return tuple(packed), valid, lengths, max_length


def unpack_kv_row(
    output_kv: tuple,
    local_row: int,
    old_length: int,
    uncached_length: int,
    max_old_length: int,
    max_uncached_length: int,
) -> tuple:
    """丢弃填充与MTP预测窗口，只保留该行的真实cache。"""
    uncached_start = max_old_length + max_uncached_length - uncached_length
    uncached_end = max_old_length + max_uncached_length
    result = []
    for key, value in output_kv:
        key_parts, value_parts = [], []
        if old_length:
            key_parts.append(key[local_row : local_row + 1, :, max_old_length - old_length : max_old_length, :])
            value_parts.append(value[local_row : local_row + 1, :, max_old_length - old_length : max_old_length, :])
        if uncached_length:
            key_parts.append(key[local_row : local_row + 1, :, uncached_start:uncached_end, :])
            value_parts.append(value[local_row : local_row + 1, :, uncached_start:uncached_end, :])
        result.append((torch.cat(key_parts, dim=2).contiguous(), torch.cat(value_parts, dim=2).contiguous()))
    return tuple(result)


def _safe_prefill_mask(input_ids: torch.Tensor, key_valid: torch.Tensor) -> torch.Tensor:
    """为左填充prompt构造无全mask行的4D可见性mask。"""
    batch, query_length = input_ids.shape
    key_index = torch.arange(query_length, device=input_ids.device).view(1, 1, query_length)
    query_index = torch.arange(query_length, device=input_ids.device).view(1, query_length, 1)
    visible = key_valid.to(dtype=torch.bool)[:, None, :] & (key_index <= query_index)
    missing_rows, missing_queries = torch.where(~visible.any(dim=-1))
    first_valid_key = key_valid.argmax(dim=-1)
    visible[missing_rows, missing_queries, first_valid_key[missing_rows]] = True
    return visible[:, None, :, :]


def _encode_visual_features(
    model: Any,
    inputs: list[BatchInput],
    dtype: torch.dtype,
    *,
    visual_batching: bool = False,
) -> list[torch.Tensor]:
    """执行MoonViT；可选将相同image_grid_hws的图片打包为TND多段序列。"""

    def grid_tensor(item: BatchInput) -> torch.Tensor:
        grid = item.image_grid_hws
        if isinstance(grid, np.ndarray):
            return torch.from_numpy(grid).to(item.pixel_values.device, dtype=torch.int32)
        if grid is None:
            raise ValueError("MoonViT批处理需要image_grid_hws")
        return grid.to(device=item.pixel_values.device, dtype=torch.int32)

    if not visual_batching or len(inputs) <= 1:
        features = []
        for item in inputs:
            visual = model.extract_feature(item.pixel_values.to(dtype), grid_tensor(item))
            features.append(model.mlp1(torch.cat(visual, dim=0)))
        return features

    groups: dict[tuple[int, ...], list[int]] = {}
    grids = []
    for index, item in enumerate(inputs):
        grid = grid_tensor(item)
        grids.append(grid)
        key = tuple(int(value) for value in grid.detach().cpu().reshape(-1).tolist())
        groups.setdefault(key, []).append(index)

    features: list[torch.Tensor | None] = [None] * len(inputs)
    for indices in groups.values():
        chunks, current, current_tokens = [], [], 0
        for index in indices:
            tokens = int(inputs[index].pixel_values.shape[0])
            if current and current_tokens + tokens > VISUAL_TND_TOKEN_LIMIT:
                chunks.append(current)
                current, current_tokens = [], 0
            current.append(index)
            current_tokens += tokens
        if current:
            chunks.append(current)
        for chunk in chunks:
            pixels = torch.cat([inputs[index].pixel_values.to(dtype) for index in chunk], dim=0)
            packed_grids = torch.cat([grids[index] for index in chunk], dim=0)
            visual_rows = model.extract_feature(pixels, packed_grids)
            if len(visual_rows) != len(chunk):
                raise RuntimeError(f"MoonViT返回{len(visual_rows)}行特征，期望{len(chunk)}行")
            lengths = [int(row.shape[0]) for row in visual_rows]
            projected = model.mlp1(torch.cat(visual_rows, dim=0))
            for index, row in zip(chunk, projected.split(lengths, dim=0)):
                features[index] = row
    if any(row is None for row in features):
        raise RuntimeError("MoonViT批处理未覆盖所有输入")
    return [row for row in features if row is not None]


def _prefill_prompt_rows(
    model: Any,
    prompt_ids: list[torch.Tensor],
    visual_features: list[torch.Tensor],
    image_token_id: int,
    pad_token_id: int,
    device: torch.device,
    static_cache: ExpandableStaticKVCache | PagedKVCache | None = None,
    slot_indices: list[int] | None = None,
    fork_paged_step: bool = False,
) -> list[tuple] | None:
    """一次批量prefill所有prompt，返回已移除填充的逐行KV cache。"""
    lengths = [int(ids.numel()) for ids in prompt_ids]
    max_length = max(lengths)
    batch = len(prompt_ids)
    input_ids = torch.full((batch, max_length), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch, max_length), dtype=torch.long, device=device)
    position_ids = torch.ones((batch, max_length), dtype=torch.long, device=device)
    for row, ids in enumerate(prompt_ids):
        length = lengths[row]
        left = max_length - length
        input_ids[row, left:] = ids.to(device)
        attention_mask[row, left:] = 1
        position_ids[row, left:] = torch.arange(length, dtype=torch.long, device=device)

    combined_visual = torch.cat(visual_features, dim=0)
    image_tokens = int((input_ids == image_token_id).sum().item())
    if image_tokens != int(combined_visual.shape[0]):
        raise RuntimeError(f"prompt中的image token数{image_tokens}与视觉特征数{combined_visual.shape[0]}不一致")
    past_key_values = None
    if static_cache is not None:
        if slot_indices is None or len(slot_indices) != batch:
            raise ValueError("静态KV prefill的slot_indices与batch大小不一致")
        slot_tensor = torch.as_tensor(slot_indices, dtype=torch.long, device=device)
        if isinstance(static_cache, PagedKVCache):
            static_cache.configure_step(
                slot_tensor,
                [0] * batch,
                position_ids,
                attention_mask,
                lengths,
                use_paged_attention=False,
                slot_indices_host=slot_indices,
            )
            if fork_paged_step:
                static_cache = static_cache.fork_current_step()
        else:
            static_cache.configure_step(slot_tensor, [0] * batch, position_ids, attention_mask, max(lengths))
        past_key_values = static_cache
    outputs = model.language_model.model(
        input_ids=input_ids,
        visual_features=combined_visual,
        image_token_index=image_token_id,
        attention_mask=_safe_prefill_mask(input_ids, attention_mask),
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=True,
        return_dict=True,
    )
    if static_cache is not None:
        return None
    return [unpack_kv_row(outputs.past_key_values, row, 0, lengths[row], 0, max_length) for row in range(batch)]


def _forward_decode(
    model: Any,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    past_key_values: tuple | None,
    logits_count: int,
    batch_bucket: int | None = None,
    npu_graph: bool = False,
) -> SimpleNamespace:
    """先运行基础decoder，再只对所需的尾部token计算lm_head。"""
    real_batch = input_ids.shape[0]
    batch_bucket = real_batch if batch_bucket is None else int(batch_bucket)
    if batch_bucket < real_batch:
        raise ValueError(f"batch_bucket={batch_bucket}不能小于真实batch={real_batch}")
    if batch_bucket > real_batch:
        padding_rows = batch_bucket - real_batch
        query_length = input_ids.shape[1]
        input_ids = torch.cat((input_ids, input_ids.new_zeros((padding_rows, query_length))), dim=0)
        dummy_mask = attention_mask.new_zeros((padding_rows, attention_mask.shape[1]))
        dummy_mask[:, -query_length:] = 1
        attention_mask = torch.cat((attention_mask, dummy_mask), dim=0)
        position_ids = torch.cat((position_ids, position_ids.new_zeros((padding_rows, query_length))), dim=0)
        if past_key_values is not None:
            padded_kv = []
            for key, value in past_key_values:
                shape = (padding_rows, *key.shape[1:])
                padded_kv.append((torch.cat((key, key.new_zeros(shape))), torch.cat((value, value.new_zeros(shape)))))
            past_key_values = tuple(padded_kv)
    graphed = None
    if npu_graph:
        from .npu_graph import run_decode_graph

        graphed = run_decode_graph(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            logits_count=logits_count,
        )
    if graphed is not None:
        logits, output_kv = graphed
        return SimpleNamespace(logits=logits[:real_batch], past_key_values=output_kv)
    if (
        isinstance(past_key_values, PagedKVCache)
        and past_key_values.use_paged_attention
        and past_key_values.direct_decode
        and not model.training
        and not torch.is_grad_enabled()
        and input_ids.device.type == "npu"
    ):
        return _forward_paged_decoder(
            model,
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            logits_count=logits_count,
            real_batch=real_batch,
        )
    outputs = model.language_model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=True,
        return_dict=True,
    )
    logits = model.language_model.lm_head(outputs.last_hidden_state[:real_batch, -logits_count:, :])
    return SimpleNamespace(logits=logits, past_key_values=outputs.past_key_values)


def _forward_paged_decoder(
    model: Any,
    *,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    past_key_values: PagedKVCache,
    logits_count: int,
    real_batch: int | None = None,
) -> SimpleNamespace:
    """直接执行paged decode所需模块，跳过固定revision中最终不会使用的4D mask构造。"""
    decoder = model.language_model.model
    hidden_states = decoder.get_input_embeddings()(input_ids)
    for decoder_layer in decoder.layers:
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=None,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=False,
            use_cache=True,
        )[0]
    hidden_states = decoder.norm(hidden_states)
    real_batch = input_ids.shape[0] if real_batch is None else int(real_batch)
    logits = model.language_model.lm_head(hidden_states[:real_batch, -logits_count:, :])
    return SimpleNamespace(logits=logits, past_key_values=past_key_values)


def _pad_generated(rows: list[list[int]], image_token_id: int, device: torch.device) -> torch.Tensor:
    """在Host完成左填充后一次搬到设备，避免逐样本H2D传输。"""
    max_length = max(len(row) for row in rows)
    output = torch.full((len(rows), max_length), image_token_id, dtype=torch.long)
    for index, row in enumerate(rows):
        output[index, max_length - len(row) :] = torch.as_tensor(row, dtype=torch.long)
    return output.to(device)


def _apply_repetition_penalty(logits: torch.Tensor, generated: torch.Tensor, penalty: float) -> torch.Tensor:
    """整批使用gather/scatter应用repetition penalty，避免逐样本Python循环和Host同步。"""
    if penalty == 1.0:
        return logits
    vocabulary = logits.shape[-1]
    tokens = generated.clamp(0, vocabulary - 1)[:, None, :].expand(-1, logits.shape[1], -1)
    selected = logits.gather(-1, tokens)
    selected = torch.where(selected > 0, selected / penalty, selected * penalty)
    logits.scatter_(-1, tokens, selected)
    return logits


def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """按未来token位置分块执行nucleus filtering，降低排序工作区峰值。"""
    for position in range(logits.shape[1]):
        values = logits[:, position, :]
        sorted_values, sorted_indices = torch.sort(values, descending=True)
        cumulative = torch.cumsum(torch.softmax(sorted_values, dim=-1), dim=-1)
        remove_sorted = cumulative > top_p
        remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
        remove_sorted[..., 0] = False
        remove = torch.zeros_like(values, dtype=torch.bool)
        remove.scatter_(-1, sorted_indices, remove_sorted)
        values.masked_fill_(remove, torch.finfo(values.dtype).min)
    return logits


def _summarize_mtp_distribution(
    values: torch.Tensor,
    token_ids: dict[str, int],
    *,
    input_is_logits: bool,
) -> PatternProbabilities:
    """把完整词表分布压缩为hybrid pattern状态机所需的少量统计。"""
    if input_is_logits:
        logits = values.float()
        log_norm = torch.logsumexp(logits, dim=-1)

        def token_probability(position: int, token: int) -> torch.Tensor:
            return torch.exp(logits[:, position, token] - log_norm[:, position])

        coord_values, coord_ids = torch.topk(logits[:, 1:5], k=4, dim=-1)
        coord_probs = torch.exp(coord_values - log_norm[:, 1:5, None])
        _, ref_ids = torch.topk(logits[:, 1:], k=5, dim=-1)
    else:
        probabilities = values

        def token_probability(position: int, token: int) -> torch.Tensor:
            return probabilities[:, position, token]

        coord_probs, coord_ids = torch.topk(probabilities[:, 1:5], k=4, dim=-1)
        _, ref_ids = torch.topk(probabilities[:, 1:], k=5, dim=-1)

    box_end = token_ids["box_end_token_id"]
    null_token = token_ids["null_token_id"]
    im_end = token_ids["im_end_token_id"]
    legal_frame = sum(token_probability(5, token) for token in (box_end, null_token, im_end))
    return PatternProbabilities(
        box_start=token_probability(0, token_ids["box_start_token_id"]),
        none=token_probability(1, token_ids["none_token_id"]),
        box_end=token_probability(2, box_end),
        null_3=token_probability(3, null_token),
        null_4=token_probability(4, null_token),
        legal_frame=legal_frame,
        coord_probs=coord_probs,
        coord_ids=coord_ids,
        ref_start=token_probability(0, token_ids["ref_start_token_id"]),
        ref_ids=ref_ids,
    )


def _candidate_top_p_values(
    logits: torch.Tensor,
    top_p: float,
    candidate_size: int = CANDIDATE_TOP_P_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """用全词表归一化精确判断Top-K候选是否完整覆盖top-p集合。"""
    candidate_size = min(int(candidate_size), logits.shape[-1])
    values, ids = torch.topk(logits, candidate_size, dim=-1, largest=True, sorted=True)
    log_normalizer = torch.logsumexp(logits, dim=-1).float()
    full_probabilities = torch.exp(values.float() - log_normalizer[..., None])
    cumulative = full_probabilities.cumsum(dim=-1)
    keep = (cumulative - full_probabilities) <= float(top_p)
    coverage = cumulative[..., -1]
    return values, ids, keep, coverage


def _sample_probabilities(
    model: Any,
    logits: torch.Tensor,
    temperature: float,
    top_p: float | None,
    generators: list[torch.Generator],
    global_rows: list[int],
    *,
    probability_mode: str = "full",
    token_ids: dict[str, int] | None = None,
    qsample_reservoir: QSampleReservoir | None = None,
    sample_slots: list[int] | None = None,
    candidate_top_p: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | PatternProbabilities | None]:
    if probability_mode not in {"full", "pattern", "none"}:
        raise ValueError(f"未知probability_mode={probability_mode!r}")
    if probability_mode == "pattern" and token_ids is None:
        raise ValueError("pattern概率压缩需要token_ids")
    if temperature > 0:
        logits.div_(max(float(temperature), 1e-8))
    bounds = torch.finfo(logits.dtype)
    logits.nan_to_num_(nan=bounds.min, posinf=bounds.max, neginf=bounds.min)
    npu_result = _sample_probabilities_npu(
        model,
        logits,
        temperature,
        top_p,
        generators,
        global_rows,
        probability_mode=probability_mode,
        token_ids=token_ids,
        qsample_reservoir=qsample_reservoir,
        sample_slots=sample_slots,
        candidate_top_p=candidate_top_p,
    )
    if npu_result is not None:
        return npu_result
    if top_p is not None and top_p < 1:
        _top_p_filter(logits, top_p)
    probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
    probabilities.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0).clamp_min_(0.0)
    row_sum = probabilities.sum(dim=-1, keepdim=True)
    bad = (~torch.isfinite(row_sum)) | (row_sum <= 0)
    if bool(bad.any().item()):
        fallback = torch.zeros_like(probabilities)
        fallback.scatter_(-1, logits.argmax(dim=-1, keepdim=True), 1.0)
        probabilities = torch.where(bad, fallback, probabilities)
        row_sum = probabilities.sum(dim=-1, keepdim=True)
    probabilities.div_(row_sum.clamp_min(1e-20))
    sampled = probabilities.argmax(dim=-1)
    if temperature > 0:
        for local_row, global_row in enumerate(global_rows):
            try:
                sampled[local_row] = torch.multinomial(
                    probabilities[local_row], 1, generator=generators[global_row]
                ).squeeze(-1)
            except RuntimeError:
                pass
    if probability_mode == "none":
        return sampled, None
    if probability_mode == "pattern":
        return sampled, _summarize_mtp_distribution(probabilities, token_ids, input_is_logits=False)
    return sampled, probabilities


def _sample_probabilities_npu(
    model: Any,
    logits: torch.Tensor,
    temperature: float,
    top_p: float | None,
    generators: list[torch.Generator],
    global_rows: list[int],
    *,
    probability_mode: str,
    token_ids: dict[str, int] | None,
    qsample_reservoir: QSampleReservoir | None,
    sample_slots: list[int] | None,
    candidate_top_p: bool,
) -> tuple[torch.Tensor, torch.Tensor | PatternProbabilities | None] | None:
    """使用单次NPU top-p qSample完成整个MTP/AR组，保留逐图片独立RNG。"""
    from ultralytics.models.locateanything.npu_fast import npu_fast_path_enabled

    if logits.device.type != "npu" or not npu_fast_path_enabled(model):
        return None
    try:
        import torch_npu
    except ImportError:
        return None
    if not hasattr(torch_npu, "npu_top_k_top_p_sample"):
        return None

    top_k_value = 0
    if (
        candidate_top_p
        and probability_mode == "pattern"
        and token_ids is not None
        and temperature > 0
        and top_p is not None
        and top_p < 1.0
    ):
        lm_head = getattr(getattr(model, "language_model", None), "lm_head", None)
        lm_weight = getattr(lm_head, "weight", None)
        signature = (
            float(top_p),
            logits.shape[-1],
            logits.device.type,
            logits.device.index,
            id(lm_weight),
            getattr(lm_weight, "_version", -1),
        )
        state = getattr(model, "_locate_candidate_top_p_state", None)
        if state is None or state[2] != signature or state[0] >= CANDIDATE_TOP_P_RECHECK_STEPS:
            _, _, _, coverage = _candidate_top_p_values(logits, top_p)
            # 首次及低频复检保留1e-4覆盖余量；避免每个decode step都因.item()同步Host。
            safe = bool((coverage >= min(float(top_p) + 1e-4, 1.0)).all().item())
            state = (0, safe, signature)
        if state[1]:
            top_k_value = min(CANDIDATE_TOP_P_SIZE, logits.shape[-1])
        object.__setattr__(model, "_locate_candidate_top_p_state", (state[0] + 1, state[1], signature))

    batch, future, vocabulary = logits.shape
    flat_logits = logits.reshape(batch * future, vocabulary).contiguous()
    nucleus = 1.0 if top_p is None else float(top_p)
    top_k, top_p_tensor = _sample_parameter_tensors(
        batch * future, top_k_value, nucleus, flat_logits.dtype, logits.device
    )
    q = None
    if temperature > 0:
        if qsample_reservoir is not None:
            if sample_slots is None:
                raise RuntimeError("qSample随机数池缺少sample_slots")
            q = qsample_reservoir.take(future, sample_slots, generators, global_rows, vocabulary)
        if q is None:
            q = _fill_q_sample_buffer(logits, future, generators, global_rows)
    sampled, filtered_logits = torch_npu.npu_top_k_top_p_sample(
        flat_logits,
        top_k,
        top_p_tensor,
        q=q,
        is_need_logits=probability_mode != "none",
        input_is_logits=True,
        post_sample="qSample",
    )
    sampled = sampled.reshape(batch, future)
    if probability_mode == "none":
        return sampled, None
    filtered_logits = filtered_logits.reshape(batch, future, vocabulary)
    if probability_mode == "pattern":
        return sampled, _summarize_mtp_distribution(filtered_logits, token_ids, input_is_logits=True)
    probabilities = torch.softmax(filtered_logits, dim=-1, dtype=torch.float32)
    return sampled, probabilities


def _fill_q_sample_buffer(
    logits: torch.Tensor,
    future: int,
    generators: list[torch.Generator],
    global_rows: list[int],
) -> torch.Tensor:
    """复用qSample指数随机缓冲区，同时维持每张图片独立RNG。"""
    vocabulary = logits.shape[-1]
    required_rows = len(global_rows) * future
    buffer_key = (logits.device.type, logits.device.index or 0, vocabulary)
    q = _Q_SAMPLE_BUFFERS.get(buffer_key)
    if q is None or q.shape[0] < required_rows:
        q = torch.empty((required_rows, vocabulary), dtype=torch.float32, device=logits.device)
        _Q_SAMPLE_BUFFERS.clear()
        _Q_SAMPLE_BUFFERS[buffer_key] = q
    q = q[:required_rows]
    for local_row, global_row in enumerate(global_rows):
        q[local_row * future : (local_row + 1) * future].exponential_(1.0, generator=generators[global_row])
    return q


def _decode_mtp_tokens(
    sampled: torch.Tensor, probabilities: torch.Tensor | PatternProbabilities, token_ids: dict[str, int]
) -> list[list[int]]:
    """批量解码bbox/ref，并只用一次D2H传输取回所有MTP token。"""
    batch = sampled.shape[0]
    coord_start = token_ids["coord_start_token_id"]
    coord_end = token_ids["coord_end_token_id"]
    box_start = token_ids["box_start_token_id"]
    box_end = token_ids["box_end_token_id"]
    none_token = token_ids["none_token_id"]
    null_token = token_ids["null_token_id"]
    ref_start = token_ids["ref_start_token_id"]

    summary = (
        probabilities
        if isinstance(probabilities, PatternProbabilities)
        else _summarize_mtp_distribution(probabilities, token_ids, input_is_logits=False)
    )
    empty_box = (
        (summary.box_start >= 0.7)
        & (summary.none > 0.2)
        & (summary.box_end > 0.2)
        & (summary.null_3 > 0.1)
        & (summary.null_4 > 0.1)
    )
    legal_frame = summary.legal_frame >= 0.2

    coord_probs, coord_ids = summary.coord_probs, summary.coord_ids
    coord_mask = (coord_ids >= coord_start) & (coord_ids <= coord_end)
    coord_valid = coord_mask.any(dim=-1).all(dim=-1)
    first_coord = coord_mask.to(torch.int64).argmax(dim=-1, keepdim=True)
    first_coord_probs = coord_probs.gather(-1, first_coord).squeeze(-1)
    first_coord_ids = coord_ids.gather(-1, first_coord).squeeze(-1)
    valid_counts = coord_mask.sum(dim=-1)
    valid_max = torch.where(coord_mask, coord_ids, -999999).max(dim=-1).values
    valid_min = torch.where(coord_mask, coord_ids, 999999).min(dim=-1).values
    abnormal = (first_coord_probs < 0.9) & (valid_counts > 1) & ((valid_max - valid_min) > 60)
    final_coords = torch.where(abnormal, 0, first_coord_ids)
    box_tokens = torch.cat(
        (
            sampled.new_full((batch, 1), box_start),
            final_coords,
            sampled.new_full((batch, 1), box_end),
        ),
        dim=-1,
    )
    empty_tokens = sampled.new_tensor((box_start, none_token, box_end, null_token, null_token, null_token)).expand(
        batch, -1
    )
    valid_box = legal_frame & coord_valid
    box_tokens = torch.where(empty_box[:, None], empty_tokens, box_tokens)
    valid_box |= empty_box

    ref_ids = summary.ref_ids
    ref_mask = (ref_ids < coord_start) | (ref_ids > coord_end)
    valid_ref = (summary.ref_start >= 0.6) & ref_mask.any(dim=-1).all(dim=-1)
    first_ref = ref_mask.to(torch.int64).argmax(dim=-1, keepdim=True)
    ref_tokens = torch.cat((sampled.new_full((batch, 1), ref_start), ref_ids.gather(-1, first_ref).squeeze(-1)), dim=-1)

    selected = torch.where(valid_ref[:, None], ref_tokens, sampled)
    selected = torch.where(valid_box[:, None], box_tokens, selected)
    return selected.cpu().tolist()


def _handle_pattern_tokens(tokens: list[int], token_ids: dict[str, int]) -> dict[str, Any]:
    """纯Python实现固定revision的hybrid pattern状态转换。"""
    null_token = token_ids["null_token_id"]
    im_end = token_ids["im_end_token_id"]
    box_start = token_ids["box_start_token_id"]
    box_end = token_ids["box_end_token_id"]
    none_token = token_ids["none_token_id"]
    coord_start = token_ids["coord_start_token_id"]
    coord_end = token_ids["coord_end_token_id"]
    ref_end = token_ids["ref_end_token_id"]
    if tokens[0] in {null_token, im_end}:
        return {"type": "im_end", "tokens": [im_end]}
    if tokens[:2] == [box_start, none_token]:
        return {"type": "empty_box", "tokens": [box_start, none_token, box_end]}
    if tokens[0] == box_start:
        coord_index = 1
        for coordinate in tokens[1:5]:
            if coord_start <= coordinate <= coord_end:
                coord_index += 1
            else:
                break
        if coord_index == 5 and tokens[5] == box_end:
            return {"type": "coord_box", "tokens": tokens}
        if coord_index == 3 and tokens[3] == box_end:
            return {"type": "point_box", "tokens": tokens[:4]}
        return {"type": "error_box", "tokens": tokens[:coord_index]}
    try:
        tokens = tokens[: tokens.index(null_token)]
    except ValueError:
        pass
    if len(tokens) >= 2 and tokens[-1] == tokens[-2] == ref_end:
        tokens = tokens[:-1]
    return {"type": "ref_object", "tokens": tokens}


def _guard_duplicate_box_pattern(
    pattern: dict[str, Any],
    row: int,
    last_patterns: list[tuple[int, ...] | None],
    repeat_counts: list[int],
    limit: int,
    im_end_token_id: int,
) -> tuple[dict[str, Any], bool]:
    """连续重复完全相同的box时终止退化生成，普通ref/box序列会重置计数。"""
    if limit <= 0:
        return pattern, False
    if pattern["type"] not in {"coord_box", "point_box", "empty_box"}:
        last_patterns[row], repeat_counts[row] = None, 0
        return pattern, False
    key = tuple(int(token) for token in pattern["tokens"])
    repeat_counts[row] = repeat_counts[row] + 1 if last_patterns[row] == key else 1
    last_patterns[row] = key
    if repeat_counts[row] <= limit:
        return pattern, False
    return {"type": "im_end", "tokens": [int(im_end_token_id)]}, True


def _sample_mtp(
    model: Any,
    logits: torch.Tensor,
    generated: torch.Tensor,
    token_ids: dict[str, int],
    temperature: float,
    top_p: float | None,
    repetition_penalty: float,
    generators: list[torch.Generator],
    rows: list[int],
    qsample_reservoir: QSampleReservoir | None = None,
    sample_slots: list[int] | None = None,
    candidate_top_p: bool = True,
) -> list[dict[str, Any]]:
    logits = _apply_repetition_penalty(logits, generated, repetition_penalty)
    sampled, probabilities = _sample_probabilities(
        model,
        logits,
        temperature,
        top_p,
        generators,
        rows,
        probability_mode="pattern",
        token_ids=token_ids,
        qsample_reservoir=qsample_reservoir,
        sample_slots=sample_slots,
        candidate_top_p=candidate_top_p,
    )
    return [
        _handle_pattern_tokens(tokens, token_ids) for tokens in _decode_mtp_tokens(sampled, probabilities, token_ids)
    ]


def _classify_ar(token: int, token_ids: dict[str, int]) -> str:
    if token == token_ids["box_end_token_id"]:
        return "box_end_ar"
    if token_ids["coord_start_token_id"] <= token <= token_ids["coord_end_token_id"]:
        return "coord_ar"
    if token == token_ids["none_token_id"]:
        return "coord_ar"
    return "im_end"


def _commit_mtp_patterns(
    patterns: list[dict[str, Any]],
    rows: list[int],
    full_ids: list[list[int]],
    generated_ids: list[list[int]],
    modes: list[str],
    finished: list[bool],
    total_limits: list[int],
    token_ids: dict[str, int],
    last_box_patterns: list[tuple[int, ...] | None] | None,
    duplicate_box_counts: list[int] | None,
    stopped_repetition: list[bool] | None,
    max_duplicate_boxes: int,
) -> list[list[int]]:
    """提交MTP状态转换，供独立和混合decoder forward共用。"""
    committed = []
    for pattern, row in zip(patterns, rows):
        if last_box_patterns is not None and duplicate_box_counts is not None:
            pattern, stopped = _guard_duplicate_box_pattern(
                pattern,
                row,
                last_box_patterns,
                duplicate_box_counts,
                max_duplicate_boxes,
                token_ids["im_end_token_id"],
            )
            if stopped and stopped_repetition is not None:
                stopped_repetition[row] = True
        accepted = [int(token) for token in pattern["tokens"]]
        committed.append(accepted)
        generated_ids[row].extend(accepted)
        full_ids[row].extend(accepted)
        if pattern["type"] == "im_end":
            finished[row] = True
        elif pattern["type"] == "error_box":
            modes[row] = "ar"
        if len(full_ids[row]) >= total_limits[row]:
            finished[row] = True
    return committed


def _commit_ar_tokens(
    tokens: list[int],
    rows: list[int],
    full_ids: list[list[int]],
    generated_ids: list[list[int]],
    modes: list[str],
    finished: list[bool],
    total_limits: list[int],
    token_ids: dict[str, int],
) -> list[list[int]]:
    """提交AR状态转换，供独立和混合decoder forward共用。"""
    for token, row in zip(tokens, rows):
        generated_ids[row].append(token)
        full_ids[row].append(token)
        state = _classify_ar(token, token_ids)
        if state == "im_end":
            finished[row] = True
        elif state == "box_end_ar":
            modes[row] = "mtp"
        if len(full_ids[row]) >= total_limits[row]:
            finished[row] = True
    return [[int(token)] for token in tokens]


def _step_mtp(
    model: Any,
    prompt_ids: list[torch.Tensor],
    kv_rows: list[tuple | None],
    rows: list[int],
    cached_lengths: list[int],
    full_ids: list[list[int]],
    generated_ids: list[list[int]],
    modes: list[str],
    finished: list[bool],
    total_limits: list[int],
    pad_token_id: int,
    mask_token_id: int,
    image_token_id: int,
    temperature: float,
    top_p: float | None,
    repetition_penalty: float,
    generators: list[torch.Generator],
    device: torch.device,
    last_box_patterns: list[tuple[int, ...] | None] | None = None,
    duplicate_box_counts: list[int] | None = None,
    stopped_repetition: list[bool] | None = None,
    max_duplicate_boxes: int = 0,
    static_cache: ExpandableStaticKVCache | PagedKVCache | None = None,
    row_slots: list[int] | None = None,
    shape_bucketing: bool = False,
    slot_capacity: int | None = None,
    kv_bucket_size: int = 128,
    npu_graph: bool = False,
    repetition_history: DeviceTokenHistory | None = None,
    qsample_reservoir: QSampleReservoir | None = None,
    candidate_top_p: bool = True,
) -> None:
    if static_cache is None:
        packed_kv, key_valid, old_lengths, max_old = pack_kv_rows(
            kv_rows,
            rows,
            device,
            length_multiple=kv_bucket_size if shape_bucketing else 1,
        )
    else:
        old_lengths = [cached_lengths[row] for row in rows]
        max_old = max(old_lengths, default=0)
        length_tensor = torch.as_tensor(old_lengths, dtype=torch.long, device=device)
        key_positions = torch.arange(max_old, device=device)[None, :]
        key_valid = (key_positions >= max_old - length_tensor[:, None]).long()
        packed_kv = static_cache
    uncached_lengths = [len(full_ids[row]) - cached_lengths[row] for row in rows]
    max_uncached = max(uncached_lengths)
    query_length = max_uncached + N_FUTURE
    metadata = torch.zeros((5, len(rows), query_length), dtype=torch.long)
    suffix, positions, query_valid, persist_mask, cache_positions = metadata.unbind(0)
    suffix.fill_(pad_token_id)
    positions.fill_(1)
    required_lengths = []
    for local_row, row in enumerate(rows):
        uncached = full_ids[row][cached_lengths[row] :]
        left = max_uncached - len(uncached)
        if uncached:
            suffix[local_row, left : left + len(uncached)] = torch.as_tensor(uncached)
            positions[local_row, left : left + len(uncached)] = torch.arange(cached_lengths[row], len(full_ids[row]))
            query_valid[local_row, left : left + len(uncached)] = 1
            persist_mask[local_row, left : left + len(uncached)] = 1
        current_length = len(full_ids[row])
        suffix[local_row, max_uncached] = full_ids[row][-1]
        positions[local_row, max_uncached] = current_length - 1
        query_valid[local_row, max_uncached] = 1
        suffix[local_row, max_uncached + 1 :] = mask_token_id
        positions[local_row, max_uncached + 1 :] = torch.arange(current_length, current_length + N_FUTURE - 1)
        query_valid[local_row, max_uncached + 1 :] = 1
        valid_count = uncached_lengths[local_row] + N_FUTURE
        start = query_length - valid_count
        cache_positions[local_row, start:] = torch.arange(old_lengths[local_row], old_lengths[local_row] + valid_count)
        required_lengths.append(old_lengths[local_row] + valid_count)
    suffix, positions, query_valid, persist_mask, cache_positions = metadata.to(device).unbind(0)

    if static_cache is not None:
        if row_slots is None:
            raise RuntimeError("静态KV decode缺少row_slots")
        slot_tensor = torch.as_tensor([row_slots[row] for row in rows], dtype=torch.long, device=device)
        if isinstance(static_cache, PagedKVCache):
            static_cache.configure_step(
                slot_tensor,
                old_lengths,
                cache_positions,
                query_valid,
                required_lengths,
                use_paged_attention=True,
                slot_indices_host=[row_slots[row] for row in rows],
            )
        else:
            static_cache.configure_step(
                slot_tensor,
                old_lengths,
                positions,
                persist_mask,
                max(len(full_ids[row]) for row in rows),
            )

    outputs = _forward_decode(
        model,
        input_ids=suffix,
        attention_mask=torch.cat((key_valid, query_valid), dim=1),
        position_ids=positions,
        past_key_values=packed_kv,
        logits_count=N_FUTURE,
        batch_bucket=(
            _ceil_bucket(len(rows), DECODE_BATCH_BUCKETS, slot_capacity or len(rows))
            if shape_bucketing and static_cache is None
            else None
        ),
        npu_graph=npu_graph,
    )
    for local_row, row in enumerate(rows):
        if static_cache is None:
            kv_rows[row] = unpack_kv_row(
                outputs.past_key_values,
                local_row,
                old_lengths[local_row],
                uncached_lengths[local_row],
                max_old,
                max_uncached,
            )
        cached_lengths[row] = len(full_ids[row])

    generated = (
        repetition_history.select([row_slots[row] for row in rows])
        if repetition_history is not None and row_slots is not None
        else _pad_generated([full_ids[row] for row in rows], image_token_id, device)
    )
    patterns = _sample_mtp(
        model,
        outputs.logits,
        generated,
        model.token_ids,
        temperature,
        top_p,
        repetition_penalty,
        generators,
        rows,
        qsample_reservoir,
        [row_slots[row] for row in rows] if qsample_reservoir is not None and row_slots is not None else None,
        candidate_top_p,
    )
    committed = _commit_mtp_patterns(
        patterns,
        rows,
        full_ids,
        generated_ids,
        modes,
        finished,
        total_limits,
        model.token_ids,
        last_box_patterns,
        duplicate_box_counts,
        stopped_repetition,
        max_duplicate_boxes,
    )
    if repetition_history is not None:
        if row_slots is None:
            raise RuntimeError("设备端repetition history缺少row_slots")
        repetition_history.append([row_slots[row] for row in rows], committed)


def _step_ar(
    model: Any,
    prompt_ids: list[torch.Tensor],
    kv_rows: list[tuple | None],
    rows: list[int],
    cached_lengths: list[int],
    full_ids: list[list[int]],
    generated_ids: list[list[int]],
    modes: list[str],
    finished: list[bool],
    total_limits: list[int],
    pad_token_id: int,
    image_token_id: int,
    temperature: float,
    top_p: float | None,
    repetition_penalty: float,
    generators: list[torch.Generator],
    device: torch.device,
    static_cache: ExpandableStaticKVCache | PagedKVCache | None = None,
    row_slots: list[int] | None = None,
    shape_bucketing: bool = False,
    slot_capacity: int | None = None,
    kv_bucket_size: int = 128,
    npu_graph: bool = False,
    repetition_history: DeviceTokenHistory | None = None,
    qsample_reservoir: QSampleReservoir | None = None,
) -> None:
    if static_cache is None:
        packed_kv, key_valid, old_lengths, max_old = pack_kv_rows(
            kv_rows,
            rows,
            device,
            length_multiple=kv_bucket_size if shape_bucketing else 1,
        )
    else:
        old_lengths = [cached_lengths[row] for row in rows]
        max_old = max(old_lengths, default=0)
        length_tensor = torch.as_tensor(old_lengths, dtype=torch.long, device=device)
        key_positions = torch.arange(max_old, device=device)[None, :]
        key_valid = (key_positions >= max_old - length_tensor[:, None]).long()
        packed_kv = static_cache
    uncached_lengths = [len(full_ids[row]) - cached_lengths[row] for row in rows]
    if any(length <= 0 for length in uncached_lengths):
        raise RuntimeError(f"AR状态没有待写入KV cache的token：rows={rows}")
    max_uncached = max(uncached_lengths)
    metadata = torch.zeros((4, len(rows), max_uncached), dtype=torch.long)
    suffix, positions, query_valid, cache_positions = metadata.unbind(0)
    suffix.fill_(pad_token_id)
    positions.fill_(1)
    required_lengths = []
    for local_row, row in enumerate(rows):
        uncached = full_ids[row][cached_lengths[row] :]
        left = max_uncached - len(uncached)
        suffix[local_row, left:] = torch.as_tensor(uncached)
        positions[local_row, left:] = torch.arange(cached_lengths[row], len(full_ids[row]))
        query_valid[local_row, left:] = 1
        cache_positions[local_row, left:] = torch.arange(old_lengths[local_row], old_lengths[local_row] + len(uncached))
        required_lengths.append(old_lengths[local_row] + len(uncached))
    suffix, positions, query_valid, cache_positions = metadata.to(device).unbind(0)
    if static_cache is not None:
        if row_slots is None:
            raise RuntimeError("静态KV decode缺少row_slots")
        slot_tensor = torch.as_tensor([row_slots[row] for row in rows], dtype=torch.long, device=device)
        if isinstance(static_cache, PagedKVCache):
            static_cache.configure_step(
                slot_tensor,
                old_lengths,
                cache_positions,
                query_valid,
                required_lengths,
                use_paged_attention=True,
                slot_indices_host=[row_slots[row] for row in rows],
            )
        else:
            static_cache.configure_step(
                slot_tensor,
                old_lengths,
                positions,
                query_valid,
                max(len(full_ids[row]) for row in rows),
            )
    outputs = _forward_decode(
        model,
        input_ids=suffix,
        attention_mask=torch.cat((key_valid, query_valid), dim=1),
        position_ids=positions,
        past_key_values=packed_kv,
        logits_count=1,
        batch_bucket=(
            _ceil_bucket(len(rows), DECODE_BATCH_BUCKETS, slot_capacity or len(rows))
            if shape_bucketing and static_cache is None
            else None
        ),
        npu_graph=npu_graph,
    )
    for local_row, row in enumerate(rows):
        if static_cache is None:
            kv_rows[row] = unpack_kv_row(
                outputs.past_key_values,
                local_row,
                old_lengths[local_row],
                uncached_lengths[local_row],
                max_old,
                max_uncached,
            )
        cached_lengths[row] = len(full_ids[row])
    generated = (
        repetition_history.select([row_slots[row] for row in rows])
        if repetition_history is not None and row_slots is not None
        else _pad_generated([full_ids[row] for row in rows], image_token_id, device)
    )
    logits = _apply_repetition_penalty(outputs.logits, generated, repetition_penalty)
    sampled, _ = _sample_probabilities(
        model,
        logits,
        temperature,
        top_p,
        generators,
        rows,
        probability_mode="none",
        qsample_reservoir=qsample_reservoir,
        sample_slots=[row_slots[row] for row in rows]
        if qsample_reservoir is not None and row_slots is not None
        else None,
    )
    committed = _commit_ar_tokens(
        sampled[:, 0].cpu().tolist(),
        rows,
        full_ids,
        generated_ids,
        modes,
        finished,
        total_limits,
        model.token_ids,
    )
    if repetition_history is not None:
        if row_slots is None:
            raise RuntimeError("设备端repetition history缺少row_slots")
        repetition_history.append([row_slots[row] for row in rows], committed)


def _step_mixed_paged(
    model: Any,
    ar_rows: list[int],
    mtp_rows: list[int],
    cached_lengths: list[int],
    full_ids: list[list[int]],
    generated_ids: list[list[int]],
    modes: list[str],
    finished: list[bool],
    total_limits: list[int],
    pad_token_id: int,
    mask_token_id: int,
    image_token_id: int,
    temperature: float,
    top_p: float | None,
    repetition_penalty: float,
    generators: list[torch.Generator],
    device: torch.device,
    static_cache: PagedKVCache,
    row_slots: list[int],
    last_box_patterns: list[tuple[int, ...] | None] | None = None,
    duplicate_box_counts: list[int] | None = None,
    stopped_repetition: list[bool] | None = None,
    max_duplicate_boxes: int = 0,
    repetition_history: DeviceTokenHistory | None = None,
    qsample_reservoir: QSampleReservoir | None = None,
    candidate_top_p: bool = True,
) -> None:
    """在一次paged decoder forward中同时推进AR和MTP行。"""
    if not ar_rows or not mtp_rows:
        raise ValueError("混合decoder forward要求同时包含AR和MTP行")
    rows = ar_rows + mtp_rows
    old_lengths = [cached_lengths[row] for row in rows]
    uncached_lengths = [len(full_ids[row]) - cached_lengths[row] for row in rows]
    if any(length <= 0 for length in uncached_lengths[: len(ar_rows)]):
        raise RuntimeError(f"混合AR状态没有待写入KV cache的token：rows={ar_rows}")
    query_lengths = [length for length in uncached_lengths[: len(ar_rows)]] + [
        length + N_FUTURE for length in uncached_lengths[len(ar_rows) :]
    ]
    query_length = max(query_lengths)
    metadata = torch.zeros((4, len(rows), query_length), dtype=torch.long)
    suffix, positions, query_valid, cache_positions = metadata.unbind(0)
    suffix.fill_(pad_token_id)
    positions.fill_(1)
    required_lengths = []

    for local_row, row in enumerate(rows):
        uncached = full_ids[row][cached_lengths[row] :]
        left = query_length - query_lengths[local_row]
        suffix[local_row, left : left + len(uncached)] = torch.as_tensor(uncached)
        positions[local_row, left : left + len(uncached)] = torch.arange(cached_lengths[row], len(full_ids[row]))
        query_valid[local_row, left : left + len(uncached)] = 1
        if local_row >= len(ar_rows):
            future_start = left + len(uncached)
            current_length = len(full_ids[row])
            suffix[local_row, future_start] = full_ids[row][-1]
            positions[local_row, future_start] = current_length - 1
            suffix[local_row, future_start + 1 :] = mask_token_id
            positions[local_row, future_start + 1 :] = torch.arange(current_length, current_length + N_FUTURE - 1)
            query_valid[local_row, future_start:] = 1
        valid_count = query_lengths[local_row]
        cache_positions[local_row, left:] = torch.arange(old_lengths[local_row], old_lengths[local_row] + valid_count)
        required_lengths.append(old_lengths[local_row] + valid_count)
    suffix, positions, query_valid, cache_positions = metadata.to(device).unbind(0)

    max_old = max(old_lengths, default=0)
    length_tensor = torch.as_tensor(old_lengths, dtype=torch.long, device=device)
    key_positions = torch.arange(max_old, device=device)[None, :]
    key_valid = (key_positions >= max_old - length_tensor[:, None]).long()
    slot_tensor = torch.as_tensor([row_slots[row] for row in rows], dtype=torch.long, device=device)
    static_cache.configure_step(
        slot_tensor,
        old_lengths,
        cache_positions,
        query_valid,
        required_lengths,
        use_paged_attention=True,
        slot_indices_host=[row_slots[row] for row in rows],
    )
    outputs = _forward_decode(
        model,
        input_ids=suffix,
        attention_mask=torch.cat((key_valid, query_valid), dim=1),
        position_ids=positions,
        past_key_values=static_cache,
        logits_count=N_FUTURE,
    )
    for row in rows:
        cached_lengths[row] = len(full_ids[row])

    ar_count = len(ar_rows)
    ar_generated = (
        repetition_history.select([row_slots[row] for row in ar_rows])
        if repetition_history is not None
        else _pad_generated([full_ids[row] for row in ar_rows], image_token_id, device)
    )
    ar_logits = _apply_repetition_penalty(outputs.logits[:ar_count, -1:], ar_generated, repetition_penalty)
    ar_sampled, _ = _sample_probabilities(
        model,
        ar_logits,
        temperature,
        top_p,
        generators,
        ar_rows,
        probability_mode="none",
        qsample_reservoir=qsample_reservoir,
        sample_slots=[row_slots[row] for row in ar_rows] if qsample_reservoir is not None else None,
    )
    mtp_generated = (
        repetition_history.select([row_slots[row] for row in mtp_rows])
        if repetition_history is not None
        else _pad_generated([full_ids[row] for row in mtp_rows], image_token_id, device)
    )
    mtp_patterns = _sample_mtp(
        model,
        outputs.logits[ar_count:, -N_FUTURE:],
        mtp_generated,
        model.token_ids,
        temperature,
        top_p,
        repetition_penalty,
        generators,
        mtp_rows,
        qsample_reservoir,
        [row_slots[row] for row in mtp_rows] if qsample_reservoir is not None else None,
        candidate_top_p,
    )
    ar_committed = _commit_ar_tokens(
        ar_sampled[:, 0].cpu().tolist(),
        ar_rows,
        full_ids,
        generated_ids,
        modes,
        finished,
        total_limits,
        model.token_ids,
    )
    mtp_committed = _commit_mtp_patterns(
        mtp_patterns,
        mtp_rows,
        full_ids,
        generated_ids,
        modes,
        finished,
        total_limits,
        model.token_ids,
        last_box_patterns,
        duplicate_box_counts,
        stopped_repetition,
        max_duplicate_boxes,
    )
    if repetition_history is not None:
        repetition_history.append([row_slots[row] for row in ar_rows], ar_committed)
        repetition_history.append([row_slots[row] for row in mtp_rows], mtp_committed)


@torch.no_grad()
def generate_batch_hybrid(
    model: Any,
    tokenizer: Any,
    inputs: list[BatchInput],
    *,
    device: torch.device,
    dtype: torch.dtype,
    seeds: list[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    scheduler: str = "pipeline",
    slot_capacity: int | None = None,
    static_kv_cache: bool = False,
    paged_kv_cache: bool = False,
    max_duplicate_boxes: int = 0,
    input_provider: Callable[[int], tuple[list[BatchInput], list[int]]] | None = None,
    completion_callback: Callable[[list[tuple[int, BatchOutput]]], None] | None = None,
    refill_batch_size: int | None = None,
    max_provider_inputs: int | None = None,
    shape_bucketing: bool = False,
    kv_bucket_size: int = 128,
    npu_graph: bool = False,
    visual_batching: bool = False,
    direct_paged_decode: bool = True,
    device_repetition_cache: bool = True,
    qsample_reservoir: bool = False,
    overlap_prefill: bool = True,
    candidate_top_p: bool = True,
) -> list[BatchOutput]:
    """以固定活跃槽位执行hybrid MTP/AR，并可从provider持续补充新样本。"""
    if len(seeds) != len(inputs):
        raise ValueError("seeds数量必须与batch大小一致")
    if isinstance(max_duplicate_boxes, bool) or not isinstance(max_duplicate_boxes, int) or max_duplicate_boxes < 0:
        raise ValueError(f"max_duplicate_boxes必须是大于等于0的整数，得到{max_duplicate_boxes!r}")
    scheduler = normalize_scheduler(scheduler)
    slot_capacity = len(inputs) if slot_capacity is None else slot_capacity
    if isinstance(slot_capacity, bool) or not isinstance(slot_capacity, int) or slot_capacity < 1:
        raise ValueError(f"slot_capacity必须是大于等于1的整数，得到{slot_capacity!r}")
    if input_provider is None:
        if not inputs:
            return []
        slot_capacity = min(slot_capacity, len(inputs))
    if refill_batch_size is None:
        refill_batch_size = max(1, slot_capacity // 16)
    if (
        isinstance(refill_batch_size, bool)
        or not isinstance(refill_batch_size, int)
        or not 1 <= refill_batch_size <= slot_capacity
    ):
        raise ValueError(f"refill_batch_size必须位于[1,{slot_capacity}]，得到{refill_batch_size!r}")
    if max_provider_inputs is None:
        max_provider_inputs = len(inputs)
    if (
        isinstance(max_provider_inputs, bool)
        or not isinstance(max_provider_inputs, int)
        or max_provider_inputs < len(inputs)
    ):
        raise ValueError(f"max_provider_inputs不能小于初始输入数{len(inputs)}，得到{max_provider_inputs!r}")
    if not isinstance(shape_bucketing, bool):
        raise ValueError(f"shape_bucketing必须是bool，得到{shape_bucketing!r}")
    if isinstance(kv_bucket_size, bool) or not isinstance(kv_bucket_size, int) or kv_bucket_size < 1:
        raise ValueError(f"kv_bucket_size必须是正整数，得到{kv_bucket_size!r}")
    if not isinstance(npu_graph, bool):
        raise ValueError(f"npu_graph必须是bool，得到{npu_graph!r}")
    if not isinstance(visual_batching, bool):
        raise ValueError(f"visual_batching必须是bool，得到{visual_batching!r}")
    if not isinstance(direct_paged_decode, bool):
        raise ValueError(f"direct_paged_decode必须是bool，得到{direct_paged_decode!r}")
    if not isinstance(device_repetition_cache, bool):
        raise ValueError(f"device_repetition_cache必须是bool，得到{device_repetition_cache!r}")
    if not isinstance(qsample_reservoir, bool):
        raise ValueError(f"qsample_reservoir必须是bool，得到{qsample_reservoir!r}")
    if not isinstance(overlap_prefill, bool):
        raise ValueError(f"overlap_prefill必须是bool，得到{overlap_prefill!r}")
    if not isinstance(candidate_top_p, bool):
        raise ValueError(f"candidate_top_p必须是bool，得到{candidate_top_p!r}")
    if npu_graph and not shape_bucketing:
        raise ValueError("npu_graph要求同时启用shape_bucketing")
    from ultralytics.models.locateanything.npu_fast import npu_fast_path_enabled
    from ultralytics.models.locateanything.npu_graph import configure_npu_graph

    if static_kv_cache and paged_kv_cache:
        raise ValueError("static_kv_cache与paged_kv_cache不能同时启用")
    if input_provider is not None and static_kv_cache:
        raise ValueError("流式continuous batching暂不支持static_kv_cache，请使用paged_kv_cache")
    fast_cache_supported = device.type == "npu" and npu_fast_path_enabled(model)
    use_static_cache = bool(static_kv_cache) and fast_cache_supported
    use_paged_cache = bool(paged_kv_cache) and fast_cache_supported
    configure_npu_graph(model, npu_graph and fast_cache_supported)
    queued_inputs: list[BatchInput | None] = list(inputs)
    image_token_id = int(model.config.image_token_index)
    token_ids = model.token_ids
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else token_ids["im_end_token_id"]
    model_limit = int(getattr(tokenizer, "model_max_length", 2**31 - 1))
    prompt_ids: list[torch.Tensor] = []
    kv_rows: list[tuple | None] = []
    cached_lengths: list[int] = []
    full_ids: list[list[int]] = []
    generated_ids: list[list[int]] = []
    modes: list[str] = []
    finished: list[bool] = []
    total_limits: list[int] = []
    generators: list[torch.Generator] = []
    forward_steps: list[int] = []
    switch_to_ar: list[int] = []
    last_box_patterns: list[tuple[int, ...] | None] = []
    duplicate_box_counts: list[int] = []
    stopped_repetition: list[bool] = []
    active: list[bool] = []
    emitted: list[bool] = []
    completed_outputs: list[BatchOutput | None] = []
    stream_outputs = input_provider is not None or completion_callback is not None
    pending = 0
    row_slots: list[int] = []
    free_slots = list(range(slot_capacity))
    provider_exhausted = input_provider is None

    def append_inputs(new_inputs: list[BatchInput], new_seeds: list[int]) -> None:
        """把provider返回的输入追加到逐样本状态数组。"""
        if len(new_inputs) != len(new_seeds):
            raise ValueError("input_provider返回的输入与seed数量不一致")
        if len(queued_inputs) + len(new_inputs) > max_provider_inputs:
            raise RuntimeError(
                f"input_provider返回样本超过声明上限：{len(queued_inputs) + len(new_inputs)}>{max_provider_inputs}"
            )
        new_prompt_ids = [item.input_ids.reshape(-1).to(device) for item in new_inputs]
        queued_inputs.extend(new_inputs)
        prompt_ids.extend(new_prompt_ids)
        kv_rows.extend([None] * len(new_inputs))
        cached_lengths.extend([0] * len(new_inputs))
        new_full_ids = [ids.detach().cpu().tolist() for ids in new_prompt_ids]
        full_ids.extend(new_full_ids)
        generated_ids.extend([[] for _ in new_inputs])
        modes.extend(["mtp"] * len(new_inputs))
        finished.extend([False] * len(new_inputs))
        total_limits.extend(min(model_limit, len(ids) + max_new_tokens) for ids in new_full_ids)
        generators.extend(make_row_generators(device, new_seeds))
        forward_steps.extend([0] * len(new_inputs))
        switch_to_ar.extend([0] * len(new_inputs))
        last_box_patterns.extend([None] * len(new_inputs))
        duplicate_box_counts.extend([0] * len(new_inputs))
        stopped_repetition.extend([False] * len(new_inputs))
        active.extend([False] * len(new_inputs))
        emitted.extend([False] * len(new_inputs))
        completed_outputs.extend([None] * len(new_inputs))
        row_slots.extend([-1] * len(new_inputs))

    initial_inputs = list(queued_inputs)
    queued_inputs.clear()
    append_inputs(initial_inputs, seeds)
    static_cache = None
    repetition_history = (
        DeviceTokenHistory(slot_capacity, max(max_new_tokens, 256), image_token_id, device)
        if use_paged_cache and device_repetition_cache and repetition_penalty != 1.0
        else None
    )
    sample_reservoir = (
        QSampleReservoir(slot_capacity, 2 * N_FUTURE, device)
        if use_paged_cache and qsample_reservoir and temperature > 0
        else None
    )
    prefill_stream = (
        torch.npu.Stream(device=device)
        if overlap_prefill and input_provider is not None and use_paged_cache and device.type == "npu"
        else None
    )
    pending_prefill: tuple[list[int], list[int], Any, list[BatchInput]] | None = None

    def create_paged_cache() -> PagedKVCache:
        """根据已到达的首批prompt创建paged池。

        continuous provider在runtime启动时还没有prompt，因此不能在首次fill_slots前预先分配。
        """
        layers = len(model.language_model.model.layers)
        block_size = 128
        initial_rows = prompt_ids[:slot_capacity]
        prompt_blocks = sum(math.ceil(len(ids) / block_size) for ids in initial_rows)
        reserve_blocks = slot_capacity * 8
        maximum_length = min(model_limit + N_FUTURE, max(len(ids) for ids in initial_rows) + max_new_tokens + N_FUTURE)
        cache = PagedKVCache(
            slot_capacity,
            layers,
            block_size=block_size,
            pool_blocks=prompt_blocks + reserve_blocks,
            max_seq_length=maximum_length,
            device=device,
        )
        cache.direct_decode = direct_paged_decode
        return cache

    if use_paged_cache and prompt_ids:
        static_cache = create_paged_cache()
    elif use_static_cache:
        layers = len(model.language_model.model.layers)
        initial_capacity = max(len(ids) for ids in full_ids) + min(max_new_tokens, 1024)
        static_cache = ExpandableStaticKVCache(slot_capacity, layers, initial_capacity)

    def activate_prefilled(rows: list[int], prefilled: list[tuple] | None) -> None:
        """在prefill完成后才开放slot，保证decode不会读取未就绪的KV。"""
        for local_row, row in enumerate(rows):
            if prefilled is not None:
                kv_rows[row] = prefilled[local_row]
            cached_lengths[row] = len(full_ids[row])
            active[row] = True
            queued_inputs[row] = None

    def promote_pending_prefill(*, force: bool = False) -> bool:
        """无阻塞查询旁路prefill；没有可运行decode时用stream依赖接管等待。"""
        nonlocal pending_prefill
        if pending_prefill is None:
            return False
        rows, _, event, _ = pending_prefill
        if not force and not event.query():
            return False
        if force:
            torch.npu.current_stream(device).wait_event(event)
        activate_prefilled(rows, None)
        pending_prefill = None
        return True

    def fill_slots() -> None:
        """批量prefill等待样本；已有decode时可排入独立NPU stream。"""
        nonlocal pending, provider_exhausted, static_cache, pending_prefill
        if pending_prefill is not None:
            return
        available = len(queued_inputs) - pending
        request_count = len(free_slots) - available
        if request_count > 0 and not provider_exhausted:
            assert input_provider is not None
            provided_inputs, provided_seeds = input_provider(request_count)
            if len(provided_inputs) > request_count:
                raise RuntimeError(f"input_provider单次返回{len(provided_inputs)}个样本，超过请求数{request_count}")
            append_inputs(provided_inputs, provided_seeds)
            if len(provided_inputs) < request_count:
                provider_exhausted = True
        count = min(len(free_slots), len(queued_inputs) - pending)
        if count <= 0:
            return
        if use_paged_cache and static_cache is None:
            static_cache = create_paged_cache()
        rows = list(range(pending, pending + count))
        slots = [free_slots.pop(0) for _ in rows]
        for row, slot in zip(rows, slots):
            row_slots[row] = slot
        if repetition_history is not None:
            repetition_history.reset_slots(slots, [full_ids[row] for row in rows])
        batch_inputs = [queued_inputs[row] for row in rows]
        if any(item is None for item in batch_inputs):
            raise RuntimeError("continuous batching输入在prefill前已被释放")
        pending += count

        def prefill(*, fork_paged_step: bool) -> list[tuple] | None:
            visual_features = (
                _encode_visual_features(model, batch_inputs, dtype, visual_batching=True)
                if visual_batching
                else _encode_visual_features(model, batch_inputs, dtype)
            )
            return _prefill_prompt_rows(
                model,
                [prompt_ids[row] for row in rows],
                visual_features,
                image_token_id,
                pad_token_id,
                device,
                static_cache=static_cache,
                slot_indices=slots if static_cache is not None else None,
                fork_paged_step=fork_paged_step,
            )

        if prefill_stream is not None and any(active):
            prefill_stream.wait_stream(torch.npu.current_stream(device))
            with torch.npu.stream(prefill_stream):
                prefilled = prefill(fork_paged_step=True)
                event = torch.npu.Event()
                event.record(prefill_stream)
            if prefilled is not None:
                raise RuntimeError("异步prefill仅支持paged KV cache")
            pending_prefill = (rows, slots, event, batch_inputs)
        else:
            activate_prefilled(rows, prefill(fork_paged_step=False))

    def refill_finished_slots() -> None:
        completed = []
        for row in range(len(queued_inputs)):
            if active[row] and finished[row]:
                active[row] = False
                kv_rows[row] = None
                if isinstance(static_cache, PagedKVCache):
                    static_cache.release_slot(row_slots[row])
                if repetition_history is not None:
                    repetition_history.release_slot(row_slots[row])
                if sample_reservoir is not None:
                    sample_reservoir.release_slot(row_slots[row])
                free_slots.append(row_slots[row])
                row_slots[row] = -1
                if stream_outputs and not emitted[row]:
                    output = BatchOutput(
                        text=tokenizer.decode(generated_ids[row], skip_special_tokens=False)
                        if generated_ids[row]
                        else "",
                        output_tokens=len(generated_ids[row]),
                        forward_steps=forward_steps[row],
                        switch_to_ar=switch_to_ar[row],
                        stopped_repetition=stopped_repetition[row],
                    )
                    completed_outputs[row] = output
                    emitted[row] = True
                    completed.append((row, output))
        free_slots.sort()
        if completed and completion_callback is not None:
            completion_callback(completed)
        if input_provider is not None:
            for row, _ in completed:
                prompt_ids[row] = torch.empty(0, dtype=torch.long)
                full_ids[row].clear()
                generated_ids[row].clear()
        if pending_prefill is not None and not any(active):
            promote_pending_prefill(force=True)
        known_waiting = len(queued_inputs) - pending
        if known_waiting or not provider_exhausted:
            threshold = (
                min(refill_batch_size, known_waiting) if provider_exhausted and known_waiting else refill_batch_size
            )
            if pending_prefill is None and (not any(active) or len(free_slots) >= threshold):
                fill_slots()

    fill_slots()

    def live(mode: str) -> list[int]:
        rows = [row for row in range(len(queued_inputs)) if active[row] and not finished[row] and modes[row] == mode]
        return sorted(rows, key=row_slots.__getitem__) if sample_reservoir is not None else rows

    def run_mtp(rows: list[int]) -> None:
        before = [modes[row] for row in rows]
        _step_mtp(
            model,
            prompt_ids,
            kv_rows,
            rows,
            cached_lengths,
            full_ids,
            generated_ids,
            modes,
            finished,
            total_limits,
            pad_token_id,
            token_ids["default_mask_token_id"],
            image_token_id,
            temperature,
            top_p,
            repetition_penalty,
            generators,
            device,
            last_box_patterns,
            duplicate_box_counts,
            stopped_repetition,
            max_duplicate_boxes,
            static_cache,
            row_slots,
            shape_bucketing,
            slot_capacity,
            kv_bucket_size,
            npu_graph,
            repetition_history,
            sample_reservoir,
            candidate_top_p,
        )
        for row, old_mode in zip(rows, before):
            forward_steps[row] += 1
            if old_mode == "mtp" and modes[row] == "ar":
                switch_to_ar[row] += 1

    def run_ar(rows: list[int]) -> None:
        _step_ar(
            model,
            prompt_ids,
            kv_rows,
            rows,
            cached_lengths,
            full_ids,
            generated_ids,
            modes,
            finished,
            total_limits,
            pad_token_id,
            image_token_id,
            temperature,
            top_p,
            repetition_penalty,
            generators,
            device,
            static_cache,
            row_slots,
            shape_bucketing,
            slot_capacity,
            kv_bucket_size,
            npu_graph,
            repetition_history,
            sample_reservoir,
        )
        for row in rows:
            forward_steps[row] += 1

    def run_mixed(ar_rows: list[int], mtp_rows: list[int]) -> None:
        """合并同一调度轮次中彼此独立的AR和MTP计算。"""
        if not isinstance(static_cache, PagedKVCache):
            raise RuntimeError("混合decoder forward要求paged KV cache")
        before = {row: modes[row] for row in ar_rows + mtp_rows}
        _step_mixed_paged(
            model,
            ar_rows,
            mtp_rows,
            cached_lengths,
            full_ids,
            generated_ids,
            modes,
            finished,
            total_limits,
            pad_token_id,
            token_ids["default_mask_token_id"],
            image_token_id,
            temperature,
            top_p,
            repetition_penalty,
            generators,
            device,
            static_cache,
            row_slots,
            last_box_patterns,
            duplicate_box_counts,
            stopped_repetition,
            max_duplicate_boxes,
            repetition_history,
            sample_reservoir,
            candidate_top_p,
        )
        for row in ar_rows + mtp_rows:
            forward_steps[row] += 1
            if before[row] == "mtp" and modes[row] == "ar":
                switch_to_ar[row] += 1

    loop = 0
    held_ar_steps = 0
    waves = max(1, (max_provider_inputs + slot_capacity - 1) // slot_capacity)
    max_loops = (max_new_tokens + 2) * waves * 2
    while (
        not provider_exhausted or pending < len(queued_inputs) or any(active) or pending_prefill is not None
    ) and loop <= max_loops:
        loop += 1
        promote_pending_prefill(force=not any(active))
        if scheduler == "hold_ar":
            ar_rows, mtp_rows = live("ar"), live("mtp")
            if ar_rows and (held_ar_steps < 5 or not mtp_rows):
                run_ar(ar_rows)
                held_ar_steps += 1
            elif mtp_rows:
                run_mtp(mtp_rows)
                held_ar_steps = 0
            refill_finished_slots()
            continue

        if scheduler in {"ar_first", "pipeline", "adaptive"}:
            ar_at_start, mtp_at_start = live("ar"), live("mtp")
            if scheduler == "adaptive" and ar_at_start and 0 < len(mtp_at_start) <= 3 and held_ar_steps < 5:
                run_ar(ar_at_start)
                held_ar_steps += 1
                refill_finished_slots()
                continue
            if scheduler == "pipeline" and ar_at_start and mtp_at_start and isinstance(static_cache, PagedKVCache):
                # 第一轮同时推进已有AR/MTP；第二轮只推进刚切换模式的行，与旧pipeline的逐行状态步数一致。
                run_mixed(ar_at_start, mtp_at_start)
                held_ar_steps = 0
                switched_mtp = [row for row in ar_at_start if active[row] and not finished[row] and modes[row] == "mtp"]
                switched_ar = [row for row in mtp_at_start if active[row] and not finished[row] and modes[row] == "ar"]
                if switched_ar and switched_mtp:
                    run_mixed(switched_ar, switched_mtp)
                elif switched_ar:
                    run_ar(switched_ar)
                elif switched_mtp:
                    run_mtp(switched_mtp)
                refill_finished_slots()
                continue
            if ar_at_start:
                run_ar(ar_at_start)
            mtp_rows = live("mtp")
            if mtp_rows:
                run_mtp(mtp_rows)
                held_ar_steps = 0
            if scheduler == "pipeline" and mtp_rows:
                old_ar = set(ar_at_start)
                new_ar = [row for row in live("ar") if row not in old_ar]
                if new_ar:
                    run_ar(new_ar)
            refill_finished_slots()
            continue

        mtp_rows = live("mtp")
        if mtp_rows:
            run_mtp(mtp_rows)
        ar_rows = live("ar")
        if ar_rows:
            run_ar(ar_rows)
        refill_finished_slots()

    if not provider_exhausted or pending < len(queued_inputs) or any(active) or pending_prefill is not None:
        raise RuntimeError(
            f"continuous batching超过安全循环上限：pending={pending}/{len(queued_inputs)} "
            f"active={sum(active)} loops={loop}"
        )

    for row, output in enumerate(completed_outputs):
        if output is None:
            completed_outputs[row] = BatchOutput(
                text=tokenizer.decode(generated_ids[row], skip_special_tokens=False) if generated_ids[row] else "",
                output_tokens=len(generated_ids[row]),
                forward_steps=forward_steps[row],
                switch_to_ar=switch_to_ar[row],
                stopped_repetition=stopped_repetition[row],
            )
    return [output for output in completed_outputs if output is not None]


__all__ = (
    "BatchInput",
    "BatchOutput",
    "SUPPORTED_SCHEDULERS",
    "generate_batch_hybrid",
    "make_row_generators",
    "normalize_scheduler",
    "pack_kv_rows",
    "unpack_kv_row",
)
