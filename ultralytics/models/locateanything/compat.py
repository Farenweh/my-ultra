# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""LocateAnything固定上游版本的Transformers兼容加载。"""

from __future__ import annotations

import inspect
import json
import sys
from importlib import metadata
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from ultralytics.utils import LOGGER

DEFAULT_MODEL = "nvidia/LocateAnything-3B"
SUPPORTED_REVISION = "c32291ca5e996f5a7a485845b4f57a233936bba0"
SUPPORTED_TRANSFORMERS = "5.14.1"
LOCATE_REQUIREMENTS = (
    f"transformers=={SUPPORTED_TRANSFORMERS}",
    "accelerate>=1.14.0",
    "peft>=0.19.1",
)


def check_locate_requirements() -> None:
    """验证LocateAnything可选依赖，避免不兼容版本产生隐晦异常。"""
    missing = []
    for package in ("transformers", "accelerate", "peft"):
        try:
            metadata.version(package)
        except metadata.PackageNotFoundError:
            missing.append(package)
    if missing:
        raise ImportError(
            f"LocateAnything缺少可选依赖：{', '.join(missing)}。请安装`pip install -e '.[locateanything]'`。"
        )
    version = metadata.version("transformers")
    if version != SUPPORTED_TRANSFORMERS:
        raise ImportError(
            f"LocateAnything当前仅验证transformers=={SUPPORTED_TRANSFORMERS}，检测到{version}。"
            "请安装`pip install -e '.[locateanything]'`。"
        )


def resolve_dtype(dtype: str | torch.dtype | None, device: torch.device) -> torch.dtype:
    """将用户dtype参数规范为torch.dtype。"""
    if isinstance(dtype, torch.dtype):
        return dtype
    name = "auto" if dtype is None else str(dtype).lower().replace("torch.", "")
    if name == "auto":
        return torch.float32 if device.type == "cpu" else torch.bfloat16
    values = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if name not in values:
        raise ValueError(f"不支持dtype={dtype!r}，可选auto、float32、float16、bfloat16")
    return values[name]


def patch_transformers_514(model_class: type, config: Any) -> None:
    """为固定revision的上游类应用Transformers 5.14兼容补丁。"""
    from transformers import PreTrainedModel
    from transformers.generation import GenerationMixin

    module_prefix = model_class.__module__.rsplit(".", 1)[0]
    dynamic_modules = {
        module_name: module
        for module_name, module in tuple(sys.modules.items())
        if module_name.startswith(module_prefix)
    }
    candidates = {model_class}
    for module in dynamic_modules.values():
        candidates.update(item for item in vars(module).values() if inspect.isclass(item))
    for candidate in candidates:
        if candidate is PreTrainedModel:
            continue
        try:
            is_model = issubclass(candidate, PreTrainedModel)
        except TypeError:
            is_model = False
        if not is_model:
            continue
        if candidate.__name__ == "Qwen2ForCausalLM" and not issubclass(candidate, GenerationMixin):
            candidate.__bases__ = (*candidate.__bases__, GenerationMixin)
        tied = candidate.__dict__.get("_tied_weights_keys")
        if tied == ["lm_head.weight"]:
            candidate._tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
        if "_check_and_adjust_attn_implementation" in candidate.__dict__:
            candidate._check_and_adjust_attn_implementation = _attention_compat_method(candidate)

    text_config = getattr(config, "text_config", None)
    rope = getattr(text_config, "rope_parameters", None)
    if text_config is not None and not hasattr(text_config, "rope_theta") and isinstance(rope, dict):
        text_config.rope_theta = rope.get("rope_theta", 10000.0)
    _patch_return_dict_aliases(config)
    _patch_legacy_attention_mask(dynamic_modules)
    _patch_generate_tensor_copy(dynamic_modules)
    _patch_moonvit_rope(dynamic_modules)
    _patch_dynamic_cache_legacy()
    _patch_composite_tied_weights(model_class)


def _patch_return_dict_aliases(config: Any) -> None:
    """在远程配置子类上恢复无弃用日志的use_return_dict只读别名。"""
    from transformers import PretrainedConfig

    for child_config in (config, getattr(config, "text_config", None), getattr(config, "vision_config", None)):
        if child_config is None or not isinstance(child_config, PretrainedConfig):
            continue
        config_class = type(child_config)
        if config_class.__dict__.get("_ultralytics_return_dict_compat", False):
            continue

        def use_return_dict(self):
            return self.return_dict

        config_class.use_return_dict = property(use_return_dict)
        config_class._ultralytics_return_dict_compat = True


def _prepare_4d_causal_attention_mask_compat(
    attention_mask: torch.Tensor | None,
    input_shape: torch.Size | tuple | list,
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    sliding_window: int | None = None,
) -> torch.Tensor | None:
    """使用Transformers 5 masking API构造远程Qwen所需的加性因果mask。"""
    from transformers.masking_utils import causal_mask_function, sdpa_mask, sliding_window_causal_mask_function

    batch_size, query_length = int(input_shape[0]), int(input_shape[-1])
    key_value_length = query_length + int(past_key_values_length)
    expected_shape = (batch_size, 1, query_length, key_value_length)
    if attention_mask is not None and attention_mask.dim() == 4:
        if tuple(attention_mask.shape) != expected_shape:
            raise ValueError(
                f"Incorrect 4D attention_mask shape: {tuple(attention_mask.shape)}; expected: {expected_shape}."
            )
        allowed = attention_mask.to(dtype=torch.bool)
    else:
        if attention_mask is None and query_length == 1 and sliding_window is None:
            return None
        mask_function = (
            sliding_window_causal_mask_function(sliding_window) if sliding_window is not None else causal_mask_function
        )
        allowed = sdpa_mask(
            batch_size=batch_size,
            q_length=query_length,
            kv_length=key_value_length,
            q_offset=past_key_values_length,
            mask_function=mask_function,
            attention_mask=attention_mask,
            local_size=sliding_window,
            allow_is_causal_skip=False,
            allow_torch_fix=False,
            device=inputs_embeds.device,
        )
        allowed = allowed.to(dtype=torch.bool)
    additive = torch.zeros(expected_shape, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
    return additive.masked_fill(~allowed, torch.finfo(inputs_embeds.dtype).min)


def _patch_legacy_attention_mask(dynamic_modules: dict[str, ModuleType]) -> None:
    """把远程Qwen对已弃用mask helper的引用替换为Transformers 5适配器。"""
    for module_name, module in dynamic_modules.items():
        if module_name.endswith(".modeling_qwen2") and hasattr(module, "_prepare_4d_causal_attention_mask"):
            module._prepare_4d_causal_attention_mask = _prepare_4d_causal_attention_mask_compat


class _TorchTensorCopyCompat:
    """仅为固定远程生成模块修正torch.tensor(tensor)的局部代理。"""

    def __init__(self, torch_module: ModuleType) -> None:
        self._torch_module = torch_module

    def __getattr__(self, name: str):
        return getattr(self._torch_module, name)

    def tensor(self, data, *args, **kwargs):
        supported = {"dtype", "device", "requires_grad"}
        if not isinstance(data, torch.Tensor) or args or set(kwargs) - supported:
            return self._torch_module.tensor(data, *args, **kwargs)
        dtype = kwargs.pop("dtype", None)
        device = kwargs.pop("device", None)
        requires_grad = kwargs.pop("requires_grad", False)
        if kwargs:
            return self._torch_module.tensor(data, **kwargs)
        result = data.detach().clone()
        to_kwargs = {}
        if dtype is not None:
            to_kwargs["dtype"] = dtype
        if device is not None:
            to_kwargs["device"] = device
        if to_kwargs:
            result = result.to(**to_kwargs)
        return result.requires_grad_(requires_grad)


def _patch_generate_tensor_copy(dynamic_modules: dict[str, ModuleType]) -> None:
    """只替换远程generate_utils模块内的torch引用，避免污染全局torch。"""
    for module_name, module in dynamic_modules.items():
        if not module_name.endswith(".generate_utils"):
            continue
        module_torch = getattr(module, "torch", None)
        if module_torch is not None and not isinstance(module_torch, _TorchTensorCopyCompat):
            module.torch = _TorchTensorCopyCompat(module_torch)


def _moonvit_rope_precompute_compat(self, device: torch.device) -> torch.Tensor:
    """以预分配ND输出构造MoonViT二维RoPE缓存，避免NPU隐式格式分配。"""
    size = self.max_height * self.max_width
    flat_pos = torch.arange(size, dtype=torch.float32, device=device)
    if device.type == "npu":
        x_pos, y_pos = torch.empty_like(flat_pos), torch.empty_like(flat_pos)
        torch.remainder(flat_pos, self.max_width, out=x_pos)
        torch.floor_divide(flat_pos, self.max_width, out=y_pos)
    else:
        x_pos = flat_pos % self.max_width
        y_pos = flat_pos // self.max_width
    dim_range = torch.arange(0, self.dim, 4, dtype=torch.float32, device=device)[: self.dim // 4]
    freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))
    x_freqs = torch.outer(x_pos, freqs)
    y_freqs = torch.outer(y_pos, freqs)
    x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
    y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
    freqs_cis = torch.cat([x_cis.unsqueeze(-1), y_cis.unsqueeze(-1)], dim=-1)
    return freqs_cis.reshape(self.max_height, self.max_width, -1)


def _patch_moonvit_rope(dynamic_modules: dict[str, ModuleType]) -> None:
    """替换会在NPU中触发internal-format警告的MoonViT二维RoPE构造。"""
    for module_name, module in dynamic_modules.items():
        if not module_name.endswith(".modeling_vit"):
            continue
        rope_class = getattr(module, "Rope2DPosEmb", None)
        if rope_class is not None and not getattr(rope_class, "_ultralytics_npu_rope_compat", False):
            rope_class._precompute_freqs_cis = _moonvit_rope_precompute_compat
            rope_class._ultralytics_npu_rope_compat = True


def _patch_processor_class(processor_class: type) -> None:
    """移除远程Processor已弃用的硬编码AutoImageProcessor声明。"""
    if processor_class.__dict__.get("image_processor_class") == "AutoImageProcessor":
        delattr(processor_class, "image_processor_class")


def _attention_compat_method(target_class: type):
    """构造只转发父类已声明关键字的注意力兼容方法。"""

    def compatible(self, attn_implementation, is_init_check=False, **kwargs):
        if attn_implementation == "magi":
            return "magi"
        method = super(target_class, self)._check_and_adjust_attn_implementation
        parameters = inspect.signature(method).parameters
        values = {"is_init_check": is_init_check, **kwargs}
        return method(attn_implementation, **{key: value for key, value in values.items() if key in parameters})

    return compatible


def _patch_composite_tied_weights(model_class: type) -> None:
    """为未调用post_init的上游组合模型补齐Transformers 5所需的tied-weight元数据。"""
    if getattr(model_class, "_ultralytics_tied_weight_compat", False):
        return
    original_init = model_class.__init__

    def compatible_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = self.get_expanded_tied_weights_keys(all_submodels=True)
        self.tie_weights(recompute_mapping=False)

    model_class.__init__ = compatible_init
    model_class._ultralytics_tied_weight_compat = True


def _patch_dynamic_cache_legacy() -> None:
    """恢复上游生成循环仍使用的Transformers 4.x DynamicCache转换接口。"""
    from transformers.cache_utils import DynamicCache

    if not hasattr(DynamicCache, "from_legacy_cache"):

        @classmethod
        def from_legacy_cache(cls, past_key_values=None):
            return cls(past_key_values)

        DynamicCache.from_legacy_cache = from_legacy_cache
    if not hasattr(DynamicCache, "to_legacy_cache"):

        def to_legacy_cache(self):
            return tuple((layer.keys, layer.values) for layer in self.layers)

        DynamicCache.to_legacy_cache = to_legacy_cache


def _repair_qwen_rope_buffers(model: torch.nn.Module) -> int:
    """重建Transformers 5低内存加载时被meta初始化为零的远程Qwen RoPE缓冲区。"""
    repaired = 0
    shared_buffers: dict[tuple[int, int, float, torch.dtype, torch.device], tuple[torch.Tensor, ...]] = {}
    for module in model.modules():
        if module.__class__.__name__ not in {"Qwen2RotaryEmbedding", "Qwen3RotaryEmbedding"}:
            continue
        required = ("dim", "base", "max_position_embeddings", "inv_freq", "cos_cached", "sin_cached")
        if not all(hasattr(module, name) for name in required):
            continue
        dtype = module.cos_cached.dtype
        device = module.cos_cached.device
        key = (int(module.dim), int(module.max_position_embeddings), float(module.base), dtype, device)
        if key not in shared_buffers:
            dim, max_length, base, _, _ = key
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
            positions = torch.arange(max_length, device=device, dtype=inv_freq.dtype)
            frequencies = torch.outer(positions, inv_freq)
            embedding = torch.cat((frequencies, frequencies), dim=-1)
            shared_buffers[key] = (inv_freq, embedding.cos().to(dtype), embedding.sin().to(dtype))
        module.inv_freq, module.cos_cached, module.sin_cached = shared_buffers[key]
        module.max_seq_len_cached = int(module.max_position_embeddings)
        repaired += 1
    return repaired


def load_locate_components(
    model_source: str | Path = DEFAULT_MODEL,
    *,
    revision: str = SUPPORTED_REVISION,
    device: torch.device,
    dtype: torch.dtype,
    attn_implementation: str = "sdpa",
    local_files_only: bool = False,
    npu_fast_path: str | bool | None = "auto",
) -> tuple[Any, Any, Any]:
    """加载固定revision的model、processor和tokenizer。"""
    check_locate_requirements()
    if revision != SUPPORTED_REVISION:
        raise ValueError(
            f"未验证LocateAnything revision {revision!r}；当前仅支持{SUPPORTED_REVISION}，以避免远程代码漂移。"
        )
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from transformers.processing_utils import ProcessorMixin

    requested_source = str(model_source)
    manifest_path = Path(requested_source) / "locateanything.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
    if manifest and manifest.get("artifact") not in {"full", "lora"}:
        raise ValueError(f"LocateAnything产物清单artifact非法：{manifest.get('artifact')!r}")
    code_source = str(manifest["base_model"]) if manifest else requested_source
    weight_source = requested_source if manifest and manifest.get("artifact") == "full" else code_source
    if manifest:
        revision = manifest.get("revision", revision)
        if revision != SUPPORTED_REVISION:
            raise ValueError(f"产物清单引用了未验证的LocateAnything revision：{revision!r}")
    config = AutoConfig.from_pretrained(
        code_source,
        trust_remote_code=True,
        revision=revision,
        local_files_only=local_files_only,
    )
    class_reference = config.auto_map["AutoModel"]
    model_class = get_class_from_dynamic_module(
        class_reference,
        code_source,
        revision=revision,
        code_revision=revision,
        local_files_only=local_files_only,
    )
    patch_transformers_514(model_class, config)
    for child_config in (config, getattr(config, "text_config", None), getattr(config, "vision_config", None)):
        if child_config is not None:
            child_config._attn_implementation = attn_implementation
            child_config._attn_implementation_autoset = False

    LOGGER.info(f"正在加载LocateAnything模型{weight_source}到{device}...")
    model = model_class.from_pretrained(
        weight_source,
        config=config,
        revision=revision,
        trust_remote_code=True,
        local_files_only=local_files_only,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    repaired_rope = _repair_qwen_rope_buffers(model)
    if repaired_rope:
        LOGGER.debug(f"已重建{repaired_rope}个LocateAnything Qwen RoPE缓冲区")
    language_model = getattr(model, "language_model", None)
    if language_model is not None and getattr(language_model.config, "tie_word_embeddings", False):
        language_model.tie_weights()
        if language_model.get_output_embeddings().weight is not language_model.get_input_embeddings().weight:
            raise RuntimeError("LocateAnything未能保留language embedding与lm-head共享权重")
    model.to(device)
    model.eval()
    _install_optional_media_stubs()
    processor_dict, _ = ProcessorMixin.get_processor_dict(
        code_source,
        trust_remote_code=True,
        revision=revision,
        local_files_only=local_files_only,
    )
    processor_reference = processor_dict["auto_map"]["AutoProcessor"]
    processor_class = get_class_from_dynamic_module(
        processor_reference,
        code_source,
        revision=revision,
        code_revision=revision,
        local_files_only=local_files_only,
    )
    _patch_processor_class(processor_class)
    processor_class.register_for_auto_class()
    processor = processor_class.from_pretrained(
        code_source,
        trust_remote_code=True,
        revision=revision,
        local_files_only=local_files_only,
        fix_mistral_regex=True,
    )
    if manifest and manifest.get("artifact") == "lora":
        from peft import PeftModel
        from safetensors.torch import load_file

        model.language_model = PeftModel.from_pretrained(model.language_model, Path(requested_source) / "llm_adapter")
        vision_adapter = Path(requested_source) / "vision_adapter"
        if vision_adapter.is_dir():
            model.vision_model = PeftModel.from_pretrained(model.vision_model, vision_adapter)
        connector = load_file(Path(requested_source) / "connector.safetensors", device="cpu")
        model.mlp1.load_state_dict(connector)
        model.to(device).eval()
    if device.type == "npu":
        from .npu_fast import install_npu_fast_path

        state = install_npu_fast_path(model, npu_fast_path)
        if state.enabled:
            LOGGER.info(
                "LocateAnything NPU快路径已启用："
                f"Qwen attention={state.attention_layers}, RMSNorm={state.rms_norm_layers}, "
                f"MoonViT attention={state.vision_attention}, RoPE={state.vision_rope}"
            )
    return model, processor, processor.tokenizer


def _install_optional_media_stubs() -> None:
    """为首版不支持的LMDB/视频依赖提供延迟报错占位，兼容aarch64环境。"""
    import importlib.util

    for name, feature in (("lmdb", "LMDB图片"), ("decord", "视频")):
        if name in sys.modules or importlib.util.find_spec(name) is not None:
            continue

        module = _missing_media_module(name, feature)
        module.__spec__ = ModuleSpec(name, loader=None)
        sys.modules[name] = module


def _missing_media_module(name: str, feature: str) -> ModuleType:
    """创建捕获独立错误信息的可选模块占位。"""

    class MissingMediaModule(ModuleType):
        def __getattr__(self, attribute):
            raise ImportError(f"当前LocateAnything首版不支持{feature}输入，因此未安装可选包{name!r}")

    return MissingMediaModule(name)


__all__ = (
    "DEFAULT_MODEL",
    "LOCATE_REQUIREMENTS",
    "SUPPORTED_REVISION",
    "SUPPORTED_TRANSFORMERS",
    "check_locate_requirements",
    "load_locate_components",
    "patch_transformers_514",
    "resolve_dtype",
)
