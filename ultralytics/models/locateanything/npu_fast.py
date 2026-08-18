# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""LocateAnything固定revision的Ascend NPU推理快路径。

快路径仅在eval + no-grad、910B、FP16/BF16且形状满足TorchNPU文档约束时启用；
训练、CUDA、CPU或不支持的输入仍调用固定revision原始实现。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch

NPU_FAST_POLICIES = frozenset({"auto", "off", "strict"})
_torch_npu = None


@dataclass(frozen=True)
class NpuFastPathState:
    """已安装的NPU快路径元数据。"""

    policy: str
    attention_layers: int
    rms_norm_layers: int
    vision_attention: bool
    vision_rope: bool

    @property
    def enabled(self) -> bool:
        return self.policy != "off" and bool(self.attention_layers or self.rms_norm_layers or self.vision_attention)


def normalize_npu_fast_policy(value: str | bool | None) -> str:
    """规范NPU快路径策略。"""
    if isinstance(value, bool):
        return "auto" if value else "off"
    policy = "auto" if value is None else str(value).strip().lower()
    aliases = {"on": "auto", "true": "auto", "1": "auto", "false": "off", "0": "off"}
    policy = aliases.get(policy, policy)
    if policy not in NPU_FAST_POLICIES:
        raise ValueError(f"npu_fast_path必须是{sorted(NPU_FAST_POLICIES)}之一，得到{value!r}")
    return policy


def _get_torch_npu():
    """只在真正选择NPU算子时导入torch_npu。"""
    global _torch_npu
    if _torch_npu is None:
        import torch_npu

        _torch_npu = torch_npu
    return _torch_npu


@lru_cache(maxsize=None)
def _supported_device(device_index: int) -> bool:
    """缓存当前NPU是否为已实测的Atlas A2产品。"""
    try:
        return "910B" in torch.npu.get_device_name(device_index).upper()
    except (AttributeError, RuntimeError):
        return False


def _jit_disabled() -> bool:
    try:
        return torch.npu.is_jit_compile_false()
    except (AttributeError, RuntimeError):
        return False


def _strict_or_original(module: Any, reason: str, *args: Any, **kwargs: Any):
    """严格模式报错；auto模式回退原forward。"""
    if getattr(module, "_locate_npu_fast_policy", "off") == "strict":
        raise RuntimeError(f"LocateAnything NPU快路径不支持当前输入：{reason}")
    return module.__class__._locate_original_forward(module, *args, **kwargs)


def _can_use_qwen_fast_path(module: Any, hidden_states: torch.Tensor, output_attentions: bool) -> str | None:
    if getattr(module, "_locate_npu_fast_policy", "off") == "off":
        return "快路径已关闭"
    if module.training or torch.is_grad_enabled():
        return "训练或启用了梯度"
    if hidden_states.device.type != "npu" or hidden_states.dtype not in {torch.float16, torch.bfloat16}:
        return f"不支持{hidden_states.device}/{hidden_states.dtype}"
    if hidden_states.ndim != 3 or not hidden_states.numel() or output_attentions:
        return "hidden states形状或output_attentions不支持"
    if module.head_dim != 128 or module.num_heads != 16 or module.num_key_value_heads != 2:
        return "不是固定Qwen2.5-3B GQA结构"
    if not _jit_disabled() or not _supported_device(hidden_states.device.index or 0):
        return "不是已验证的910B非JIT环境"
    try:
        torch_npu = _get_torch_npu()
        if not all(
            hasattr(torch_npu, name)
            for name in ("npu_fused_infer_attention_score_v2", "npu_incre_flash_attention", "npu_rotary_mul")
        ):
            return "TorchNPU缺少attention或RoPE算子"
    except (AttributeError, ImportError, RuntimeError):
        return "torch_npu不可用"
    return None


def _blocked_attention_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
    """将PyTorch SDPA可见性/加性mask转为TorchNPU中True=屏蔽的mask。"""
    if mask is None:
        return None
    if mask.dtype == torch.bool:
        blocked = ~mask
    else:
        blocked = mask < 0
    if blocked.ndim == 4 and blocked.shape[1] == 1:
        return _to_nd(blocked.contiguous())
    raise ValueError(f"LocateAnything attention mask形状非法：{tuple(mask.shape)} {mask.dtype}")


def _to_nd(tensor: torch.Tensor) -> torch.Tensor:
    """仅在TorchNPU输入不是ND时执行格式转换。"""
    torch_npu = _get_torch_npu()
    return (
        tensor
        if torch_npu.get_npu_format(tensor) == torch_npu.Format.ND
        else torch_npu.npu_format_cast(tensor, torch_npu.Format.ND)
    )


def _apply_qwen_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """使用half RotaryMul同时处理Q/K，保留Qwen系数索引语义。"""
    cos = cos[position_ids].unsqueeze(1).to(dtype=query.dtype).contiguous()
    sin = sin[position_ids].unsqueeze(1).to(dtype=query.dtype).contiguous()
    torch_npu = _get_torch_npu()
    query = torch_npu.npu_rotary_mul(query.contiguous(), cos, sin, rotary_mode="half")
    key = torch_npu.npu_rotary_mul(key.contiguous(), cos, sin, rotary_mode="half")
    return query, key


def _fused_qkv_projection(module: Any, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """以非注册推理缓存合并Q/K/V线性投影；参数版本变化时自动重建。"""
    projections = (module.q_proj, module.k_proj, module.v_proj)
    if not all(type(projection) is torch.nn.Linear for projection in projections):
        return tuple(projection(hidden_states) for projection in projections)
    versions = tuple(
        (projection.weight._version, projection.bias._version if projection.bias is not None else -1)
        for projection in projections
    )
    cache = getattr(module, "_locate_qkv_cache", None)
    if cache is None or cache[0] != versions:
        weight = torch.cat([projection.weight for projection in projections], dim=0)
        bias = (
            torch.cat([projection.bias for projection in projections], dim=0)
            if all(projection.bias is not None for projection in projections)
            else None
        )
        cache = (versions, weight, bias)
        object.__setattr__(module, "_locate_qkv_cache", cache)
    combined = torch.nn.functional.linear(hidden_states, cache[1], cache[2])
    sizes = [projection.out_features for projection in projections]
    return combined.split(sizes, dim=-1)


def _qwen_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_value: Any = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs: Any,
):
    """不重复GQA KV heads的Qwen Prompt/Incre Flash Attention。"""
    reason = _can_use_qwen_fast_path(self, hidden_states, output_attentions)
    if reason is not None:
        return _strict_or_original(
            self,
            reason,
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )

    batch, query_length, _ = hidden_states.shape
    if getattr(self, "_locate_npu_fused_qkv", False):
        query, key, value = _fused_qkv_projection(self, hidden_states)
    else:
        query, key, value = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
    query = query.view(batch, query_length, self.num_heads, self.head_dim).transpose(1, 2)
    key = key.view(batch, query_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value = value.view(batch, query_length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    kv_length = query_length + (past_key_value.get_seq_length(self.layer_idx) if past_key_value is not None else 0)
    rope_cache = getattr(past_key_value, "_npu_rope_cache", None)
    if rope_cache is None:
        cos, sin = self.rotary_emb(value, seq_len=kv_length)
        if past_key_value is not None and hasattr(past_key_value, "_npu_rope_cache"):
            past_key_value._npu_rope_cache = (cos, sin)
    else:
        cos, sin = rope_cache
    query, key = _apply_qwen_rope(query, key, cos, sin, position_ids)
    if past_key_value is not None:
        key, value = past_key_value.update(
            key,
            value,
            self.layer_idx,
            {"sin": sin, "cos": cos},
        )
    query, key, value = (_to_nd(tensor.contiguous()) for tensor in (query, key, value))
    torch_npu = _get_torch_npu()
    scale = self.head_dim**-0.5
    paged_attention = bool(getattr(past_key_value, "use_paged_attention", False))
    blocked_mask = None if paged_attention else _blocked_attention_mask(attention_mask)
    if paged_attention:
        block_table = getattr(past_key_value, "_active_block_table_nd", None)
        if block_table is None:
            block_table = _to_nd(past_key_value.active_block_table)
            past_key_value._active_block_table_nd = block_table
    if paged_attention and query_length == 1:
        attention_output = torch_npu.npu_incre_flash_attention(
            query,
            key,
            value,
            actual_seq_lengths=past_key_value.actual_seq_lengths,
            block_table=block_table,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_key_value_heads,
            scale_value=scale,
            input_layout="BNSD",
            block_size=past_key_value.block_size,
            inner_precise=1,
        )
    elif paged_attention:
        paged_mask = getattr(past_key_value, "_paged_attention_mask_nd", None)
        if paged_mask is None:
            paged_mask = _to_nd(past_key_value.paged_attention_mask.contiguous())
            past_key_value._paged_attention_mask_nd = paged_mask
            if getattr(past_key_value, "paged_sparse_mode", 0) == 3:
                past_key_value._sparse_causal_mask_nd = paged_mask
        attention_output = torch_npu.npu_fused_infer_attention_score_v2(
            query,
            key,
            value,
            atten_mask=paged_mask,
            actual_seq_qlen=past_key_value.actual_query_lengths,
            actual_seq_kvlen=past_key_value.actual_seq_lengths,
            block_table=block_table,
            num_query_heads=self.num_heads,
            num_key_value_heads=self.num_key_value_heads,
            softmax_scale=scale,
            input_layout="BNSD",
            block_size=past_key_value.block_size,
            inner_precise=0,
            sparse_mode=getattr(past_key_value, "paged_sparse_mode", 0),
        )[0]
    elif query_length == 1:
        attention_output = torch_npu.npu_incre_flash_attention(
            query,
            key,
            value,
            atten_mask=blocked_mask,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_key_value_heads,
            scale_value=scale,
            input_layout="BNSD",
            inner_precise=1,
        )
    else:
        attention_output = torch_npu.npu_fused_infer_attention_score_v2(
            query,
            key,
            value,
            atten_mask=blocked_mask,
            num_query_heads=self.num_heads,
            num_key_value_heads=self.num_key_value_heads,
            softmax_scale=scale,
            input_layout="BNSD",
            inner_precise=0,
        )[0]
    attention_output = attention_output.transpose(1, 2).contiguous().reshape(batch, query_length, self.hidden_size)
    return self.o_proj(attention_output), None, past_key_value


def _qwen_rms_norm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """在支持的eval NPU输入上使用融合RMSNorm。"""
    supported = (
        getattr(self, "_locate_npu_fast_policy", "off") != "off"
        and not self.training
        and not torch.is_grad_enabled()
        and hidden_states.device.type == "npu"
        and hidden_states.dtype in {torch.float16, torch.bfloat16}
        and self.weight.device == hidden_states.device
        and self.weight.dtype == hidden_states.dtype
        and 2 <= hidden_states.ndim <= 8
        and _supported_device(hidden_states.device.index or 0)
    )
    try:
        supported = supported and hasattr(_get_torch_npu(), "npu_rms_norm")
    except (AttributeError, ImportError, RuntimeError):
        supported = False
    if supported:
        return _get_torch_npu().npu_rms_norm(hidden_states, self.weight, epsilon=self.variance_epsilon)[0]
    return _strict_or_original(self, "RMSNorm输入不满足融合算子约束", hidden_states)


def _fused_gate_up_projection(module: Any, hidden_states: torch.Tensor) -> torch.Tensor:
    """构造非参数Gate/Up拼接权重，并通过参数版本保证训练后不会复用旧缓存。"""
    projections = (module.gate_proj, module.up_proj)
    versions = tuple(projection.weight._version for projection in projections)
    cache = getattr(module, "_locate_gate_up_cache", None)
    if cache is None or cache[0] != versions:
        cache = (versions, torch.cat([projection.weight for projection in projections], dim=0))
        object.__setattr__(module, "_locate_gate_up_cache", cache)
    return torch.nn.functional.linear(hidden_states, cache[1])


def _qwen_mlp_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """推理时以单次Gate/Up投影和NPU SwiGLU执行Qwen MLP。"""
    supported = (
        getattr(self, "_locate_npu_fused_mlp", False)
        and getattr(self, "_locate_npu_fast_policy", "off") != "off"
        and not self.training
        and not torch.is_grad_enabled()
        and hidden_states.device.type == "npu"
        and hidden_states.dtype in {torch.float16, torch.bfloat16}
        and type(self.gate_proj) is torch.nn.Linear
        and type(self.up_proj) is torch.nn.Linear
        and _supported_device(hidden_states.device.index or 0)
    )
    try:
        supported = supported and hasattr(_get_torch_npu(), "npu_swiglu")
    except (AttributeError, ImportError, RuntimeError):
        supported = False
    if not supported:
        return self.__class__._locate_original_forward(self, hidden_states)
    projected = _fused_gate_up_projection(self, hidden_states)
    return self.down_proj(_get_torch_npu().npu_swiglu(projected, dim=-1))


def _qwen_decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_value: Any = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs: Any,
):
    """融合attention residual add与post-attention RMSNorm。"""
    supported = (
        getattr(self, "_locate_npu_fused_add_rms", True)
        and getattr(self, "_locate_npu_fast_policy", "off") != "off"
        and not self.training
        and not torch.is_grad_enabled()
        and hidden_states.device.type == "npu"
        and hidden_states.dtype in {torch.float16, torch.bfloat16}
        and _supported_device(hidden_states.device.index or 0)
    )
    try:
        supported = supported and hasattr(_get_torch_npu(), "npu_add_rms_norm")
    except (AttributeError, ImportError, RuntimeError):
        supported = False
    if not supported:
        return self.__class__._locate_original_forward(
            self,
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )

    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
    )
    normalized, _, hidden_states = _get_torch_npu().npu_add_rms_norm(
        hidden_states,
        residual,
        self.post_attention_layernorm.weight,
        self.post_attention_layernorm.variance_epsilon,
    )
    residual = hidden_states
    hidden_states = residual + self.mlp(normalized)
    outputs = (hidden_states,)
    if output_attentions:
        outputs += (self_attn_weights,)
    if use_cache:
        outputs += (present_key_value,)
    return outputs


def _vision_rope_fast(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """将MoonViT复数RoPE转为FP32 interleave RotaryMul。"""
    if (
        xq.device.type != "npu"
        or xq.ndim != 3
        or xk.shape != xq.shape
        or xq.dtype not in {torch.float16, torch.bfloat16}
        or not _supported_device(xq.device.index or 0)
    ):
        return _vision_rope_fast._locate_original(xq, xk, freqs_cis)
    coefficients = freqs_cis.unsqueeze(-2)
    cos = coefficients.real.repeat_interleave(2, dim=-1).unsqueeze(0).contiguous()
    sin = coefficients.imag.repeat_interleave(2, dim=-1).unsqueeze(0).contiguous()
    torch_npu = _get_torch_npu()
    query = torch_npu.npu_rotary_mul(xq.float().unsqueeze(0).contiguous(), cos, sin, rotary_mode="interleave").squeeze(
        0
    )
    key = torch_npu.npu_rotary_mul(xk.float().unsqueeze(0).contiguous(), cos, sin, rotary_mode="interleave").squeeze(0)
    return query.to(xq.dtype), key.to(xk.dtype)


def _vision_cumulative_lengths(cu_seqlens: torch.Tensor) -> tuple[int, ...]:
    """按cu_seqlens张量对象缓存Host累计长度，避免allocator地址复用造成串批。"""
    lengths = getattr(cu_seqlens, "_locate_vision_cumulative_lengths", None)
    if lengths is None:
        lengths = tuple(int(value) for value in cu_seqlens[1:].cpu().tolist())
        cu_seqlens._locate_vision_cumulative_lengths = lengths
    return lengths


def _vision_attention_fast(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_cu_seqlens: torch.Tensor | None = None,
    k_cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """单图使用BSND Fusion Attention，多段序列保留TND block-diagonal语义。"""
    supported = (
        query.device.type == "npu"
        and query.ndim == 3
        and query.shape == key.shape == value.shape
        and query.dtype in {torch.float16, torch.bfloat16}
        and query.shape[-1] == 72
        and q_cu_seqlens is not None
        and k_cu_seqlens is not None
        and _supported_device(query.device.index or 0)
    )
    try:
        supported = supported and hasattr(_get_torch_npu(), "npu_fusion_attention")
    except (AttributeError, ImportError, RuntimeError):
        supported = False
    if not supported:
        return _vision_attention_fast._locate_original(
            query,
            key,
            value,
            q_cu_seqlens=q_cu_seqlens,
            k_cu_seqlens=k_cu_seqlens,
        )
    # 同一个cu_seqlens对象会被全部MoonViT层复用；把Host list绑定到对象生命周期，
    # 避免NPU allocator复用data_ptr后让新batch误命中旧序列长度。
    lengths = _vision_cumulative_lengths(q_cu_seqlens)
    if not lengths or lengths[-1] != query.shape[0]:
        raise RuntimeError(
            f"MoonViT TND序列元数据不一致：query tokens={query.shape[0]}, cumulative length={lengths[-1] if lengths else 0}"
        )
    if len(lengths) == 1:
        output = (
            _get_torch_npu()
            .npu_fusion_attention_v3(
                query.unsqueeze(0).contiguous(),
                key.unsqueeze(0).contiguous(),
                value.unsqueeze(0).contiguous(),
                query.shape[-2],
                "BSND",
                scale=query.shape[-1] ** -0.5,
                keep_prob=1.0,
            )[0]
            .squeeze(0)
        )
    else:
        output = _get_torch_npu().npu_fusion_attention(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            query.shape[-2],
            "TND",
            scale=query.shape[-1] ** -0.5,
            keep_prob=1.0,
            actual_seq_qlen=lengths,
            actual_seq_kvlen=lengths,
        )[0]
    return output.flatten(start_dim=-2)


def _moonvit_attention_qkvpacked(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rope_freqs_cis: torch.Tensor | None = None,
) -> torch.Tensor:
    """以实例策略执行MoonViT融合RoPE和attention，避免污染远程模块全局函数表。"""
    supported = (
        getattr(self, "_locate_npu_fast_policy", "off") != "off"
        and not self.training
        and not torch.is_grad_enabled()
        and hidden_states.device.type == "npu"
        and hidden_states.dtype in {torch.float16, torch.bfloat16}
        and rope_freqs_cis is not None
    )
    if not supported:
        if getattr(self, "_locate_npu_fast_policy", "off") == "strict":
            raise RuntimeError("LocateAnything NPU快路径不支持当前MoonViT输入")
        return self.__class__._locate_original_attention_qkvpacked(
            self, hidden_states, cu_seqlens, rope_freqs_cis=rope_freqs_cis
        )
    query_key_value = self.wqkv(hidden_states)
    query_key_value = query_key_value.view(
        *query_key_value.shape[:-1], 3, self.num_heads, self.hidden_size_per_attention_head
    )
    query, key, value = torch.unbind(query_key_value, dim=-3)
    query, key = _vision_rope_fast(query, key, rope_freqs_cis)
    output = _vision_attention_fast(
        query,
        key,
        value,
        q_cu_seqlens=cu_seqlens,
        k_cu_seqlens=cu_seqlens,
    )
    return self.wo(output)


def _patch_class_forward(target_class: type, replacement: Any) -> None:
    """幂等替换固定revision远程类的forward。"""
    if not hasattr(target_class, "_locate_original_forward"):
        target_class._locate_original_forward = target_class.forward
        target_class.forward = replacement


def _patch_moonvit_attention(target_class: type) -> None:
    """幂等替换MoonViT实例attention入口。"""
    if not hasattr(target_class, "_locate_original_attention_qkvpacked"):
        target_class._locate_original_attention_qkvpacked = target_class.attention_qkvpacked
        target_class.attention_qkvpacked = _moonvit_attention_qkvpacked


def install_npu_fast_path(model: torch.nn.Module, policy: str | bool | None = "auto") -> NpuFastPathState:
    """为已加载的固定revision LocateAnything安装受保护NPU快路径。"""
    policy = normalize_npu_fast_policy(policy)
    if policy == "off":
        for module in model.modules():
            if hasattr(module, "_locate_npu_fast_policy"):
                module._locate_npu_fast_policy = "off"
        configure_npu_kernel_fusions(model, fused_qkv=False, fused_add_rms_norm=False)
        state = NpuFastPathState(policy, 0, 0, False, False)
        model._locate_npu_fast_state = state
        return state
    devices = {parameter.device.type for parameter in model.parameters()}
    if "npu" not in devices:
        if policy == "strict":
            raise RuntimeError("npu_fast_path='strict'要求模型已加载到Ascend NPU")
        state = NpuFastPathState(policy, 0, 0, False, False)
        model._locate_npu_fast_state = state
        return state

    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format = True

    attention_layers = 0
    rms_norm_layers = 0
    vision_attention_layers = 0
    for module in model.modules():
        name = module.__class__.__name__
        if name in {"Qwen2SdpaAttention", "Qwen2SdpaAttentionGqa"}:
            _patch_class_forward(module.__class__, _qwen_attention_forward)
            module._locate_npu_fast_policy = policy
            module._locate_npu_fused_qkv = False
            attention_layers += 1
        elif name == "Qwen2RMSNorm":
            _patch_class_forward(module.__class__, _qwen_rms_norm_forward)
            module._locate_npu_fast_policy = policy
            rms_norm_layers += 1
        elif name == "Qwen2DecoderLayer":
            _patch_class_forward(module.__class__, _qwen_decoder_layer_forward)
            module._locate_npu_fast_policy = policy
            module._locate_npu_fused_add_rms = True
        elif name == "Qwen2MLP":
            _patch_class_forward(module.__class__, _qwen_mlp_forward)
            module._locate_npu_fast_policy = policy
            module._locate_npu_fused_mlp = False
        elif name == "MoonVitEncoderLayer":
            _patch_moonvit_attention(module.__class__)
            module._locate_npu_fast_policy = policy
            vision_attention_layers += 1

    vision_attention = False
    vision_rope = False
    vision_model = getattr(model, "vision_model", None)
    if vision_model is not None:
        vision_module = __import__(type(vision_model).__module__, fromlist=["*"])
        functions = getattr(vision_module, "VL_VISION_ATTENTION_FUNCTIONS", None)
        if isinstance(functions, dict) and "sdpa" in functions:
            if not hasattr(_vision_attention_fast, "_locate_original"):
                _vision_attention_fast._locate_original = functions["sdpa"]
            vision_attention = vision_attention_layers > 0
        original_rope = getattr(vision_module, "apply_rope", None)
        if original_rope is not None:
            if not hasattr(_vision_rope_fast, "_locate_original"):
                _vision_rope_fast._locate_original = original_rope
            vision_rope = vision_attention_layers > 0

    state = NpuFastPathState(policy, attention_layers, rms_norm_layers, vision_attention, vision_rope)
    if policy == "strict" and (
        attention_layers != 36 or rms_norm_layers != 73 or vision_attention_layers != 27 or not vision_rope
    ):
        raise RuntimeError(f"LocateAnything NPU快路径安装不完整：{state}")
    model._locate_npu_fast_state = state
    return state


def configure_npu_kernel_fusions(
    model: Any,
    *,
    fused_qkv: bool = False,
    fused_add_rms_norm: bool = True,
    fused_mlp: bool = False,
) -> None:
    """配置可独立A/B的推理融合，并在关闭QKV融合时释放额外权重缓存。"""
    for module in model.modules() if hasattr(model, "modules") else ():
        name = module.__class__.__name__
        if name in {"Qwen2SdpaAttention", "Qwen2SdpaAttentionGqa"}:
            module._locate_npu_fused_qkv = bool(fused_qkv)
            if not fused_qkv:
                object.__setattr__(module, "_locate_qkv_cache", None)
        elif name == "Qwen2DecoderLayer":
            module._locate_npu_fused_add_rms = bool(fused_add_rms_norm)
        elif name == "Qwen2MLP":
            module._locate_npu_fused_mlp = bool(fused_mlp)
            if not fused_mlp:
                object.__setattr__(module, "_locate_gate_up_cache", None)


def release_npu_inference_caches(model: Any) -> None:
    """训练前释放所有非参数推理缓存，保持state_dict和可训练参数不变。"""
    configure_npu_kernel_fusions(model, fused_qkv=False, fused_add_rms_norm=False, fused_mlp=False)


def npu_fast_path_enabled(model: Any) -> bool:
    """返回当前模型是否已启用NPU快路径。"""
    state = getattr(model, "_locate_npu_fast_state", None)
    return bool(state and state.enabled)


__all__ = (
    "NPU_FAST_POLICIES",
    "NpuFastPathState",
    "install_npu_fast_path",
    "configure_npu_kernel_fusions",
    "normalize_npu_fast_policy",
    "npu_fast_path_enabled",
    "release_npu_inference_caches",
)
