from pathlib import Path
from urllib.parse import urlparse

import torch
import torch.nn as nn

from .third_party.dinov2.hubconf import (
    dinov2_vitb14,
    dinov2_vitb14_reg,
    dinov2_vitg14,
    dinov2_vitg14_reg,
    dinov2_vitl14,
    dinov2_vitl14_reg,
    dinov2_vits14,
    dinov2_vits14_reg,
)
from .third_party.dinov3.hubconf import (
    dinov3_convnext_base,
    dinov3_convnext_large,
    dinov3_convnext_small,
    dinov3_convnext_tiny,
    dinov3_vit7b16,
    dinov3_vitb16,
    dinov3_vith16plus,
    dinov3_vitl16,
    dinov3_vitl16plus,
    dinov3_vits16,
    dinov3_vits16plus,
)
from .third_party.c_radio_v3 import CRADIO_V3_CONFIGS, VisionTransformer as CRADIOv3VisionTransformer
from .third_party.pe_spatial import PE_SPATIAL_CONFIGS, VisionTransformer as PESpatialVisionTransformer


def _is_url(path: str) -> bool:
    parsed = urlparse(path)
    return parsed.scheme in {"http", "https", "file"}


def resolve_dinov3_weights(pretrained: str, filename: str) -> str:
    """Resolve a DINOv3 local weights directory or explicit URL/path to a loadable path."""
    if _is_url(pretrained):
        return pretrained
    return str((Path(pretrained).expanduser() / filename).resolve())


class SigLIP2So400M(nn.Module):
    """提供检测器友好空间输出的SigLIP 2 So400m Patch16 NaFlex视觉骨干网络。"""

    input_size_divisor = 16
    feature_stride = 16
    model_name = "naflexvit_so400m_patch16_siglip.v2_webli"
    architecture = "naflexvit_so400m_patch16_siglip"

    def __init__(self, pretrained: str | bool = True, max_num_patches: int | None = 2500):
        super().__init__()
        if max_num_patches is not None and (
            isinstance(max_num_patches, bool) or not isinstance(max_num_patches, int) or max_num_patches <= 0
        ):
            raise ValueError(f"max_num_patches必须是正整数或None，但得到的是 {max_num_patches!r}")
        self.max_num_patches = max_num_patches

        import timm

        if pretrained is True:
            self.model = timm.create_model(self.model_name, pretrained=True, num_classes=0)
        elif pretrained is False:
            self.model = timm.create_model(self.architecture, pretrained=False, num_classes=0)
        elif isinstance(pretrained, str):
            checkpoint_path = Path(pretrained).expanduser()
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"找不到SigLIP 2本地权重文件: {checkpoint_path}")
            self.model = timm.create_model(
                self.architecture,
                pretrained=False,
                num_classes=0,
            )
            timm.models.load_checkpoint(self.model, str(checkpoint_path.absolute()))
        else:
            raise TypeError(f"pretrained必须是bool或本地权重路径，但得到的是 {type(pretrained).__name__}")

    def _check_input(self, x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise AssertionError(f"SigLIP2So400M的输入形状应该为BCHW，但得到的是 {tuple(x.shape)}")
        if x.shape[1] != 3:
            raise AssertionError(f"SigLIP2So400M要求三通道输入，但得到的是 {x.shape[1]} 通道")
        if not torch.is_floating_point(x):
            raise TypeError(f"SigLIP2So400M要求浮点输入，但得到的是 {x.dtype}")

        h, w = x.shape[-2:]
        divisor = self.input_size_divisor
        if h % divisor or w % divisor:
            raise AssertionError(f"SigLIP2So400M要求输入的高度和宽度是{divisor}的倍数，但得到的是 {(h, w)}")

        num_patches = (h // divisor) * (w // divisor)
        if self.max_num_patches is not None and num_patches > self.max_num_patches:
            raise ValueError(
                f"SigLIP2So400M输入 {(h, w)} 会生成 {num_patches} 个patch，超过max_num_patches="
                f"{self.max_num_patches}；请减小imgsz、提高max_num_patches或设为None"
            )

    def _forward_intermediate(self, x: torch.Tensor, output_fmt: str) -> torch.Tensor:
        self._check_input(x)
        x = x.mul(2.0).sub(1.0)
        return self.model.forward_intermediates(
            x,
            indices=1,
            norm=True,
            output_fmt=output_fmt,
            intermediates_only=True,
        )[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_intermediate(x, output_fmt="NCHW")

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_intermediate(x, output_fmt="NLC")

    @staticmethod
    def dims() -> int:
        return 1152


class CRADIOv3(nn.Module):
    """提供检测器友好空间输出的C-RADIOv3 B/L/H/g视觉骨干网络。"""

    def __init__(self, scale: str, pretrained: str | bool = True):
        super().__init__()
        if not isinstance(scale, str):
            raise TypeError(f"scale必须是字符串，但得到的是{type(scale).__name__}")
        self.scale = scale.lower()
        if self.scale not in CRADIO_V3_CONFIGS:
            raise ValueError(f"不存在的C-RADIOv3架构预设{scale!r}，应该为{tuple(CRADIO_V3_CONFIGS)}")
        if not isinstance(pretrained, (bool, str)):
            raise TypeError(f"pretrained必须是bool或本地权重路径，但得到的是{type(pretrained).__name__}")

        config = CRADIO_V3_CONFIGS[self.scale]
        self.input_size_divisor = config.patch_size
        self.feature_stride = config.patch_size
        self.preferred_resolution = config.preferred_resolution
        self.max_resolution = config.max_resolution
        loading_weights = pretrained is not False
        self.model = CRADIOv3VisionTransformer(
            config,
            initialize=not loading_weights,
            device="meta" if loading_weights else None,
        )
        if loading_weights:
            self.model.to_empty(device="cpu")

        from timm.data.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

        self.register_buffer("norm_mean", torch.tensor(OPENAI_CLIP_MEAN).view(3, 1, 1))
        self.register_buffer("norm_std", torch.tensor(OPENAI_CLIP_STD).view(3, 1, 1))
        if pretrained is True:
            from huggingface_hub import hf_hub_download

            checkpoint = hf_hub_download(
                repo_id=config.repo_id,
                filename="model.safetensors",
                revision=config.revision,
            )
            self._load_checkpoint(checkpoint)
        elif isinstance(pretrained, str):
            checkpoint = Path(pretrained).expanduser()
            if not checkpoint.is_file():
                raise FileNotFoundError(f"找不到C-RADIOv3本地权重文件: {checkpoint}")
            self._load_checkpoint(checkpoint)

    @staticmethod
    def _map_checkpoint_key(key: str) -> str | None:
        while key.startswith("module."):
            key = key.removeprefix("module.")
        if key == "radio_model.summary_idxs":
            return None
        if key.startswith("radio_model.model."):
            return f"model.{key.removeprefix('radio_model.model.')}"
        if key == "radio_model.input_conditioner.norm_mean":
            return "norm_mean"
        if key == "radio_model.input_conditioner.norm_std":
            return "norm_std"
        if key.startswith("base_model."):
            return f"model.{key.removeprefix('base_model.')}"
        if key == "input_conditioner.norm_mean":
            return "norm_mean"
        if key == "input_conditioner.norm_std":
            return "norm_std"
        return key

    def _target_tensors(self) -> dict[str, torch.Tensor]:
        return {**dict(self.named_parameters()), **dict(self.named_buffers())}

    def _copy_checkpoint_tensors(self, keys, get_tensor) -> None:
        targets = self._target_tensors()
        mapped = {}
        ignored = []
        for source_key in keys:
            target_key = self._map_checkpoint_key(source_key)
            if target_key is None:
                ignored.append(source_key)
                continue
            if target_key in mapped:
                raise RuntimeError(f"C-RADIOv3 checkpoint中多个权重映射到同一键{target_key!r}")
            mapped[target_key] = source_key
        missing = sorted(set(targets) - set(mapped))
        unexpected = sorted(set(mapped) - set(targets))
        if missing or unexpected:
            raise RuntimeError(
                "C-RADIOv3 checkpoint与模型结构不匹配："
                f"missing_keys={missing}, unexpected_keys={unexpected}, ignored_keys={ignored}"
            )
        with torch.no_grad():
            for target_key, target in targets.items():
                source = get_tensor(mapped[target_key])
                if not isinstance(source, torch.Tensor):
                    raise TypeError(f"C-RADIOv3 checkpoint键{mapped[target_key]!r}不是Tensor")
                if source.shape != target.shape:
                    raise RuntimeError(
                        f"C-RADIOv3 checkpoint键{mapped[target_key]!r}形状不匹配："
                        f"checkpoint={tuple(source.shape)}, model={tuple(target.shape)}"
                    )
                target.copy_(source.to(dtype=target.dtype))

    def _load_checkpoint(self, checkpoint: str | Path) -> None:
        checkpoint = Path(checkpoint)
        if checkpoint.suffix == ".safetensors":
            from safetensors import safe_open

            with safe_open(checkpoint, framework="pt", device="cpu") as state:
                self._copy_checkpoint_tensors(state.keys(), state.get_tensor)
            return

        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise TypeError(f"C-RADIOv3 checkpoint必须是字典，但得到的是{type(state).__name__}")
        for container in ("state_dict_ema", "state_dict", "weights"):
            if isinstance(state.get(container), dict):
                state = state[container]
                break
        self._copy_checkpoint_tensors(state.keys(), state.__getitem__)

    def train(self, mode: bool = True):
        result = super().train(mode)
        if hasattr(self, "model"):
            self.model.frozen_deterministic = bool(mode and not any(p.requires_grad for p in self.parameters()))
        return result

    def _check_input(self, x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise AssertionError(f"CRADIOv3的输入形状应该为BCHW，但得到的是{tuple(x.shape)}")
        if x.shape[1] != 3:
            raise AssertionError(f"CRADIOv3要求三通道输入，但得到的是{x.shape[1]}通道")
        if not torch.is_floating_point(x):
            raise TypeError(f"CRADIOv3要求浮点输入，但得到的是{x.dtype}")
        height, width = x.shape[-2:]
        if height % self.feature_stride or width % self.feature_stride:
            raise AssertionError(f"CRADIOv3要求输入高度和宽度是16的倍数，但得到的是{(height, width)}")
        if height > self.max_resolution or width > self.max_resolution:
            raise ValueError(f"CRADIOv3输入最大边不能超过{self.max_resolution}，但得到的是{(height, width)}")

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        x = (x - self.norm_mean) / self.norm_std
        return self.model.forward_features(x)[:, self.model.patch_generator.num_skip :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        sequence = self.forward_sequence(x)
        grid_h, grid_w = height // self.feature_stride, width // self.feature_stride
        return sequence.transpose(1, 2).reshape(x.shape[0], self.dims(self.scale), grid_h, grid_w)

    @staticmethod
    def dims(scale: str) -> int:
        try:
            return CRADIO_V3_CONFIGS[scale.lower()].width
        except (AttributeError, KeyError) as error:
            raise ValueError(f"不存在的C-RADIOv3架构预设{scale!r}") from error

    @staticmethod
    def stride(scale: str) -> int:
        try:
            return CRADIO_V3_CONFIGS[scale.lower()].patch_size
        except (AttributeError, KeyError) as error:
            raise ValueError(f"不存在的C-RADIOv3架构预设{scale!r}") from error


class PESpatial(nn.Module):
    """提供检测器友好空间输出的 PE-Spatial T/S/B/L/G 视觉骨干网络。"""

    def __init__(self, scale: str, pretrained: str | bool = True):
        super().__init__()
        if not isinstance(scale, str):
            raise TypeError(f"scale必须是字符串，但得到的是{type(scale).__name__}")
        self.scale = scale.lower()
        if self.scale not in PE_SPATIAL_CONFIGS:
            raise ValueError(f"不存在的PE-Spatial架构预设{scale!r}，应该为{tuple(PE_SPATIAL_CONFIGS)}")
        config = PE_SPATIAL_CONFIGS[self.scale]
        self.input_size_divisor = config.patch_size
        self.feature_stride = config.patch_size
        self.model = PESpatialVisionTransformer(config)

        if pretrained is True:
            from huggingface_hub import hf_hub_download

            checkpoint = hf_hub_download(
                repo_id=f"facebook/{config.checkpoint}",
                filename=f"{config.checkpoint}.pt",
            )
            self._load_checkpoint(checkpoint)
        elif pretrained is False:
            pass
        elif isinstance(pretrained, str):
            checkpoint = Path(pretrained).expanduser()
            if not checkpoint.is_file():
                raise FileNotFoundError(f"找不到PE-Spatial本地权重文件: {checkpoint}")
            self._load_checkpoint(checkpoint)
        else:
            raise TypeError(f"pretrained必须是bool或本地权重路径，但得到的是{type(pretrained).__name__}")

    def _load_checkpoint(self, checkpoint: str | Path) -> None:
        """严格加载官方视觉 checkpoint，并兼容其历史包装前缀。"""
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise TypeError(f"PE-Spatial checkpoint必须是字典，但得到的是{type(state).__name__}")
        if isinstance(state.get("state_dict"), dict):
            state = state["state_dict"]
        elif isinstance(state.get("weights"), dict):
            state = state["weights"]

        state = {key.removeprefix("module."): value for key, value in state.items()}
        if any(key.startswith("visual.") for key in state):
            state = {key.removeprefix("visual."): value for key, value in state.items() if key.startswith("visual.")}
        incompatible = self.model.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "PE-Spatial checkpoint与模型结构不匹配："
                f"missing_keys={incompatible.missing_keys}, unexpected_keys={incompatible.unexpected_keys}"
            )

    def _check_input(self, x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise AssertionError(f"PESpatial的输入形状应该为BCHW，但得到的是{tuple(x.shape)}")
        if x.shape[1] != 3:
            raise AssertionError(f"PESpatial要求三通道输入，但得到的是{x.shape[1]}通道")
        if not torch.is_floating_point(x):
            raise TypeError(f"PESpatial要求浮点输入，但得到的是{x.dtype}")
        height, width = x.shape[-2:]
        if height % self.feature_stride or width % self.feature_stride:
            raise AssertionError(
                f"PE-Spatial-{self.scale.upper()}要求输入高度和宽度是{self.feature_stride}的倍数，"
                f"但得到的是{(height, width)}"
            )

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        x = x.mul(2.0).sub(1.0)
        return self.model.forward_features(x, norm=False, strip_cls_token=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        sequence = self.forward_sequence(x)
        grid_h, grid_w = height // self.feature_stride, width // self.feature_stride
        return sequence.transpose(1, 2).reshape(x.shape[0], self.dims(self.scale), grid_h, grid_w)

    @staticmethod
    def dims(scale: str) -> int:
        try:
            return PE_SPATIAL_CONFIGS[scale.lower()].width
        except (AttributeError, KeyError) as error:
            raise ValueError(f"不存在的PE-Spatial架构预设{scale!r}") from error

    @staticmethod
    def stride(scale: str) -> int:
        try:
            return PE_SPATIAL_CONFIGS[scale.lower()].patch_size
        except (AttributeError, KeyError) as error:
            raise ValueError(f"不存在的PE-Spatial架构预设{scale!r}") from error


class DINOv2(nn.Module):
    input_size_divisor = 14
    feature_stride = 14

    def __init__(self, scale: str, pretrained: str | bool = True):
        super().__init__()
        scaletable = {
            "s": dinov2_vits14,
            "b": dinov2_vitb14,
            "l": dinov2_vitl14,
            "g": dinov2_vitg14,
            "s_reg": dinov2_vits14_reg,
            "b_reg": dinov2_vitb14_reg,
            "l_reg": dinov2_vitl14_reg,
            "g_reg": dinov2_vitg14_reg,
        }
        self.scale = scale.lower()
        assert self.scale in scaletable.keys(), f"不存在的DINOv2架构预设，应该为{scaletable.keys()}"

        kwargs = {"pretrained": pretrained is not False}
        if isinstance(pretrained, str):
            kwargs["weights"] = pretrained
        self.model = scaletable[self.scale](**kwargs)

    def _check_input(self, x: torch.Tensor):
        if x.ndim != 4:
            raise AssertionError(f"DINOv2的输入形状应该为BCHW，但得到的是 {tuple(x.shape)}")
        h, w = x.shape[-2:]
        divisor = self.input_size_divisor
        if h % divisor or w % divisor:
            raise AssertionError(f"DINOv2要求输入的高度和宽度是{divisor}的倍数，但得到的是 {(h, w)}")

    def forward(self, x: torch.Tensor):
        self._check_input(x)
        return self.model.get_intermediate_layers(x, n=1, reshape=True)[0]

    def forward_sequence(self, x: torch.Tensor):
        self._check_input(x)
        return self.model.get_intermediate_layers(x, n=1, reshape=False)[0]

    @staticmethod
    def dims(scale: str):
        dimstable = {
            "s": 384,
            "b": 768,
            "l": 1024,
            "g": 1536,
            "s_reg": 384,
            "b_reg": 768,
            "l_reg": 1024,
            "g_reg": 1536,
        }
        return dimstable[scale.lower()]


class DINOv3ViT(nn.Module):
    input_size_divisor = 16
    feature_stride = 16

    def __init__(self, scale: str, pretrained: str | bool = "./weights", rescale_coords: None | int = None):
        super(DINOv3ViT, self).__init__()
        scaletable = {
            "s": dinov3_vits16,
            "sp": dinov3_vits16plus,
            "b": dinov3_vitb16,
            "l": dinov3_vitl16,
            "lp": dinov3_vitl16plus,
            "hp": dinov3_vith16plus,
            "7b": dinov3_vit7b16,
        }
        weightstable = {"l": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"}
        self.scale = scale.lower()
        assert self.scale in scaletable.keys(), f"不存在的DINOv3ViT架构预设，应该为{scaletable.keys()}"
        if pretrained is False:
            kv = {"pretrained": False, "weights": None}
        elif isinstance(pretrained, str):
            kv = {"pretrained": True, "weights": resolve_dinov3_weights(pretrained, weightstable[self.scale])}
        else:
            kv = {"pretrained": True}
        self.model = scaletable[self.scale](**kv)
        self.model.rope_embed.rescale_coords = rescale_coords

    def forward(self, x: torch.Tensor):
        result = self.model.get_intermediate_layers(x, n=1, reshape=True)[0]
        return result

    def forward_sequence(self, x: torch.Tensor):
        return self.model.get_intermediate_layers(x, n=1, reshape=False)[0]

    @staticmethod
    def dims(scale: str):
        dimstable = {
            "s": 384,
            "sp": 384,
            "b": 768,
            "l": 1024,
            "lp": 1024,
            "hp": 1280,
            "7b": 4096,
        }
        return dimstable[scale.lower()]
