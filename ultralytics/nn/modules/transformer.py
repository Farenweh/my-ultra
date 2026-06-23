# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Transformer modules."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import constant_, trunc_normal_, xavier_uniform_

from ultralytics.utils.checks import IS_ASCEND
from ultralytics.utils.torch_utils import TORCH_1_11

from .conv import Conv
from .third_party.dinov3.dinov3.layers import LayerScale, RopePositionEmbedding, SelfAttentionBlock
from .utils import _get_clones, multi_scale_deformable_attn_pytorch

__all__ = (
    "AIFI",
    "MLP",
    "DeformableTransformerEncoder",
    "DeformableTransformerEncoderLayer",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "LayerNorm2d",
    "MLPBlock",
    "MSDeformAttn",
    "TransformerBlock",
    "TransformerEncoderLayer",
    "TransformerLayer",
)


class TransformerEncoderLayer(nn.Module):
    """A single layer of the transformer encoder.

    This class implements a standard transformer encoder layer with multi-head attention and feedforward network,
    supporting both pre-normalization and post-normalization configurations.

    Attributes:
        ma (nn.MultiheadAttention): Multi-head attention module.
        fc1 (nn.Linear): First linear layer in the feedforward network.
        fc2 (nn.Linear): Second linear layer in the feedforward network.
        norm1 (nn.LayerNorm): Layer normalization after attention.
        norm2 (nn.LayerNorm): Layer normalization after feedforward network.
        dropout (nn.Dropout): Dropout layer for the feedforward network.
        dropout1 (nn.Dropout): Dropout layer after attention.
        dropout2 (nn.Dropout): Dropout layer after feedforward network.
        act (nn.Module): Activation function.
        normalize_before (bool): Whether to apply normalization before attention and feedforward.
    """

    def __init__(
        self,
        c1: int,
        cm: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.0,
        act: nn.Module | None = None,
        normalize_before: bool = False,
    ):
        """Initialize the TransformerEncoderLayer with specified parameters.

        Args:
            c1 (int): Input dimension.
            cm (int): Hidden dimension in the feedforward network.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout probability.
            act (nn.Module): Activation function.
            normalize_before (bool): Whether to apply normalization before attention and feedforward.
        """
        super().__init__()
        from ...utils.torch_utils import TORCH_1_9

        if not TORCH_1_9:
            raise ModuleNotFoundError(
                "TransformerEncoderLayer() requires torch>=1.9 to use nn.MultiheadAttention(batch_first=True)."
            )
        self.ma = nn.MultiheadAttention(c1, num_heads, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)

        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.act = nn.GELU() if act is None else act
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, pos: torch.Tensor | None = None) -> torch.Tensor:
        """Add position embeddings to the tensor if provided."""
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Perform forward pass with post-normalization.

        Args:
            src (torch.Tensor): Input tensor.
            src_mask (torch.Tensor, optional): Mask for the src sequence.
            src_key_padding_mask (torch.Tensor, optional): Mask for the src keys per batch.
            pos (torch.Tensor, optional): Positional encoding.

        Returns:
            (torch.Tensor): Output tensor after attention and feedforward.
        """
        q = k = self.with_pos_embed(src, pos)
        src2 = self.ma(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)

    def forward_pre(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Perform forward pass with pre-normalization.

        Args:
            src (torch.Tensor): Input tensor.
            src_mask (torch.Tensor, optional): Mask for the src sequence.
            src_key_padding_mask (torch.Tensor, optional): Mask for the src keys per batch.
            pos (torch.Tensor, optional): Positional encoding.

        Returns:
            (torch.Tensor): Output tensor after attention and feedforward.
        """
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.ma(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src2))))
        return src + self.dropout2(src2)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward propagate the input through the encoder module.

        Args:
            src (torch.Tensor): Input tensor.
            src_mask (torch.Tensor, optional): Mask for the src sequence.
            src_key_padding_mask (torch.Tensor, optional): Mask for the src keys per batch.
            pos (torch.Tensor, optional): Positional encoding.

        Returns:
            (torch.Tensor): Output tensor after transformer encoder layer.
        """
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class AIFI(TransformerEncoderLayer):
    """AIFI transformer layer for 2D data with positional embeddings.

    This class extends TransformerEncoderLayer to work with 2D feature maps by adding 2D sine-cosine positional
    embeddings and handling the spatial dimensions appropriately.
    """

    def __init__(
        self,
        c1: int,
        cm: int = 2048,
        num_heads: int = 8,
        dropout: float = 0,
        act: nn.Module | None = None,
        normalize_before: bool = False,
    ):
        """Initialize the AIFI instance with specified parameters.

        Args:
            c1 (int): Input dimension.
            cm (int): Hidden dimension in the feedforward network.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout probability.
            act (nn.Module): Activation function.
            normalize_before (bool): Whether to apply normalization before attention and feedforward.
        """
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)
        self._cached_pos_embed_key = None
        self._cached_pos_embed = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the AIFI transformer layer.

        Args:
            x (torch.Tensor): Input tensor with shape [B, C, H, W].

        Returns:
            (torch.Tensor): Output tensor with shape [B, C, H, W].
        """
        c, h, w = x.shape[1:]
        if torch.jit.is_tracing():
            pos_embed = self.build_2d_sincos_position_embedding(w, h, c, device=x.device).to(dtype=x.dtype)
        else:
            pos_key = (w, h, c)
            if self._cached_pos_embed_key != pos_key or self._cached_pos_embed is None:
                self._cached_pos_embed_key = pos_key
                self._cached_pos_embed = self.build_2d_sincos_position_embedding(w, h, c, device=x.device).to(
                    dtype=x.dtype
                )
            else:
                self._cached_pos_embed = self._cached_pos_embed.to(device=x.device, dtype=x.dtype)
            pos_embed = self._cached_pos_embed
        # Flatten [B, C, H, W] to [B, HxW, C]
        x = super().forward(x.flatten(2).permute(0, 2, 1), pos=pos_embed)
        return x.permute(0, 2, 1).view([-1, c, h, w]).contiguous()

    @staticmethod
    def build_2d_sincos_position_embedding(
        w: int, h: int, embed_dim: int = 256, temperature: float = 10000.0, device=None
    ) -> torch.Tensor:
        """Build 2D sine-cosine position embedding.

        Args:
            w (int): Width of the feature map.
            h (int): Height of the feature map.
            embed_dim (int): Embedding dimension.
            temperature (float): Temperature for the sine/cosine functions.
            device (torch.device, optional): Device on which to build the embedding grids.

        Returns:
            (torch.Tensor): Position embedding with shape [1, h*w, embed_dim].
        """
        assert embed_dim % 4 == 0, "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        # Build on the input's device so a traced graph doesn't bake a CPU `arange` that clashes with GPU activations
        # (e.g. TorchScript export of RT-DETR: the traced `arange` is pinned to CPU and fails GPU inference).
        grid_w = torch.arange(w, dtype=torch.float32, device=device)
        grid_h = torch.arange(h, dtype=torch.float32, device=device)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij") if TORCH_1_11 else torch.meshgrid(grid_w, grid_h)
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32, device=device) / pos_dim
        omega = 1.0 / (temperature**omega)

        # Pin matmul to fp32 for CoreML export: fp16 sin/cos on integer-derived positions accumulates visible error.
        out_w = grid_w.flatten()[..., None].float() @ omega[None]
        out_h = grid_h.flatten()[..., None].float() @ omega[None]

        return torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], 1)[None]


class TransformerLayer(nn.Module):
    """Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)."""

    def __init__(self, c: int, num_heads: int):
        """Initialize a self-attention mechanism using linear transformations and multi-head attention.

        Args:
            c (int): Input and output channel dimension.
            num_heads (int): Number of attention heads.
        """
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a transformer block to the input x and return the output.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after transformer layer.
        """
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        return self.fc2(self.fc1(x)) + x


class TransformerBlock(nn.Module):
    """Vision Transformer block based on https://arxiv.org/abs/2010.11929.

    This class implements a complete transformer block with optional convolution layer for channel adjustment, learnable
    position embedding, and multiple transformer layers.

    Attributes:
        conv (Conv, optional): Convolution layer if input and output channels differ.
        linear (nn.Linear): Learnable position embedding.
        tr (nn.Sequential): Sequential container of transformer layers.
        c2 (int): Output channel dimension.
    """

    def __init__(self, c1: int, c2: int, num_heads: int, num_layers: int):
        """Initialize a Transformer module with position embedding and specified number of heads and layers.

        Args:
            c1 (int): Input channel dimension.
            c2 (int): Output channel dimension.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer layers.
        """
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward propagate the input through the transformer block.

        Args:
            x (torch.Tensor): Input tensor with shape [b, c1, h, w].

        Returns:
            (torch.Tensor): Output tensor with shape [b, c2, h, w].
        """
        if self.conv is not None:
            x = self.conv(x)
        b, _, h, w = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, h, w)


class MLPBlock(nn.Module):
    """A single block of a multi-layer perceptron."""

    def __init__(self, embedding_dim: int, mlp_dim: int, act=nn.GELU):
        """Initialize the MLPBlock with specified embedding dimension, MLP dimension, and activation function.

        Args:
            embedding_dim (int): Input and output dimension.
            mlp_dim (int): Hidden dimension.
            act (type): Activation function class.
        """
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the MLPBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after MLP block.
        """
        return self.lin2(self.act(self.lin1(x)))


class MLP(nn.Module):
    """A simple multi-layer perceptron (also called FFN).

    This class implements a configurable MLP with multiple linear layers, activation functions, and optional sigmoid
    output activation.

    Attributes:
        num_layers (int): Number of layers in the MLP.
        layers (nn.ModuleList): List of linear layers.
        sigmoid (bool): Whether to apply sigmoid to the output.
        act (nn.Module): Activation function.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        act=nn.ReLU,
        sigmoid: bool = False,
        residual: bool = False,
        out_norm: nn.Module = None,
    ):
        """Initialize the MLP with specified input, hidden, output dimensions and number of layers.

        Args:
            input_dim (int): Input dimension.
            hidden_dim (int): Hidden dimension.
            output_dim (int): Output dimension.
            num_layers (int): Number of layers.
            act (type): Activation function class.
            sigmoid (bool): Whether to apply sigmoid to the output.
            residual (bool): Whether to use residual connections.
            out_norm (nn.Module, optional): Normalization layer for the output.
        """
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim, *h], [*h, output_dim]))
        self.sigmoid = sigmoid
        self.act = act()
        if residual and input_dim != output_dim:
            raise ValueError("residual is only supported if input_dim == output_dim")
        self.residual = residual
        # whether to apply a normalization layer to the output
        assert isinstance(out_norm, nn.Module) or out_norm is None
        self.out_norm = out_norm or nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the entire MLP.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after MLP.
        """
        orig_x = x
        for i, layer in enumerate(self.layers):
            x = getattr(self, "act", nn.ReLU())(layer(x)) if i < self.num_layers - 1 else layer(x)
        if getattr(self, "residual", False):
            x = x + orig_x
        x = getattr(self, "out_norm", nn.Identity())(x)
        return x.sigmoid() if getattr(self, "sigmoid", False) else x


class LayerNorm2d(nn.Module):
    """2D Layer Normalization module inspired by Detectron2 and ConvNeXt implementations.

    This class implements layer normalization for 2D feature maps, normalizing across the channel dimension while
    preserving spatial dimensions.

    Attributes:
        weight (nn.Parameter): Learnable scale parameter.
        bias (nn.Parameter): Learnable bias parameter.
        eps (float): Small constant for numerical stability.

    References:
        https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py
        https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        """Initialize LayerNorm2d with the given parameters.

        Args:
            num_channels (int): Number of channels in the input.
            eps (float): Small constant for numerical stability.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass for 2D layer normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Normalized output tensor.
        """
        u = x.mean(1, keepdim=True)
        centered = x - u
        s = (centered * centered).mean(1, keepdim=True)
        x = centered / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class MSDeformAttn(nn.Module):
    """Multiscale Deformable Attention Module based on Deformable-DETR and PaddleDetection implementations.

    This module implements multiscale deformable attention that can attend to features at multiple scales with learnable
    sampling locations and attention weights.

    Attributes:
        im2col_step (int): Step size for im2col operations.
        d_model (int): Model dimension.
        n_levels (int): Number of feature levels.
        n_heads (int): Number of attention heads.
        n_points (int): Number of sampling points per attention head per feature level.
        sampling_offsets (nn.Linear): Linear layer for generating sampling offsets.
        attention_weights (nn.Linear): Linear layer for generating attention weights.
        value_proj (nn.Linear): Linear layer for projecting values.
        output_proj (nn.Linear): Linear layer for projecting output.

    References:
        https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py
    """

    def __init__(self, d_model: int = 256, n_levels: int = 4, n_heads: int = 8, n_points: int = 4):
        """Initialize MSDeformAttn with the given parameters.

        Args:
            d_model (int): Model dimension.
            n_levels (int): Number of feature levels.
            n_heads (int): Number of attention heads.
            n_points (int): Number of sampling points per attention head per feature level.
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, but got {d_model} and {n_heads}")
        _d_per_head = d_model // n_heads
        # Better to set _d_per_head to a power of 2 which is more efficient in a CUDA implementation
        assert _d_per_head * n_heads == d_model, "`d_model` must be divisible by `n_heads`"

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self._cached_value_shapes = None
        self._cached_offset_normalizer = None

        self._reset_parameters()

    def _reset_parameters(self):
        """Reset module parameters."""
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        refer_bbox: torch.Tensor,
        value: torch.Tensor,
        value_shapes: list,
        value_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Perform forward pass for multiscale deformable attention.

        Args:
            query (torch.Tensor): Query tensor with shape [bs, query_length, C].
            refer_bbox (torch.Tensor): Reference points or boxes with shape [bs, query_length, 1, 2 or 4] for decoder
                use, or [bs, query_length, n_levels, 2] for encoder use. Values are in [0, 1] with top-left (0, 0)
                and bottom-right (1, 1). The size-1 level axis broadcasts across n_levels.
            value (torch.Tensor): Value tensor with shape [bs, value_length, C].
            value_shapes (list): List with shape [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})].
            value_mask (torch.Tensor, optional): Mask tensor with shape [bs, value_length], True for padding elements,
                False for non-padding elements.

        Returns:
            (torch.Tensor): Output tensor with shape [bs, Length_{query}, C].

        References:
            https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
        """
        bs, len_q = query.shape[:2]
        len_v = value.shape[1]
        assert sum(s[0] * s[1] for s in value_shapes) == len_v

        value = self.value_proj(value)
        if value_mask is not None:
            if IS_ASCEND and value.device.type == "npu":
                value = value * (~value_mask[..., None]).to(value.dtype)
            else:
                value = value.masked_fill(value_mask[..., None], float(0))
        value = value.view(bs, len_v, self.n_heads, self.d_model // self.n_heads)
        # Fold (n_levels, n_points) into one axis so every traced tensor stays at rank <= 5 (required for CoreML
        # export). Decoder reference points use one level axis, while encoder reference points provide n_levels.
        n_total_points = self.n_levels * self.n_points
        sampling_offsets = self.sampling_offsets(query).view(bs, len_q, self.n_heads, n_total_points, 2)
        attention_weights = self.attention_weights(query).view(bs, len_q, self.n_heads, n_total_points)
        attention_weights = F.softmax(attention_weights, -1)
        num_points = refer_bbox.shape[-1]
        if num_points == 2:
            num_reference_levels = refer_bbox.shape[-2]
            if num_reference_levels not in {1, self.n_levels}:
                raise ValueError(
                    f"Reference points must have 1 or {self.n_levels} levels, but got {num_reference_levels}."
                )
            value_shapes_tuple = tuple((int(h), int(w)) for h, w in value_shapes)
            if (
                self._cached_value_shapes != value_shapes_tuple
                or self._cached_offset_normalizer is None
                or self._cached_offset_normalizer.dtype != query.dtype
                or self._cached_offset_normalizer.device != query.device
            ):
                self._cached_value_shapes = value_shapes_tuple
                self._cached_offset_normalizer = (
                    torch.as_tensor(value_shapes_tuple, dtype=query.dtype, device=query.device)
                    .flip(-1)[:, None, :]
                    .expand(-1, self.n_points, -1)
                    .reshape(n_total_points, 2)
                )
            offset_normalizer = self._cached_offset_normalizer
            sampling_offsets = sampling_offsets / offset_normalizer
            if num_reference_levels == 1:
                sampling_locations = refer_bbox[:, :, None, :, :] + sampling_offsets
            else:
                reference_points = (
                    refer_bbox[:, :, None, :, None, :]
                    .expand(-1, -1, self.n_heads, -1, self.n_points, -1)
                    .reshape(bs, len_q, self.n_heads, n_total_points, 2)
                )
                sampling_locations = reference_points + sampling_offsets
        elif num_points == 4:
            sampling_locations = (
                refer_bbox[:, :, None, :, :2] + sampling_offsets / self.n_points * refer_bbox[:, :, None, :, 2:] * 0.5
            )
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {num_points}.")
        output = multi_scale_deformable_attn_pytorch(value, value_shapes, sampling_locations, attention_weights)
        return self.output_proj(output)


class DeformableTransformerEncoderLayer(nn.Module):
    """Single multiscale deformable transformer encoder layer."""

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_ffn: int = 1024,
        dropout: float = 0.0,
        act: nn.Module = nn.ReLU(),
        n_levels: int = 4,
        n_points: int = 4,
    ):
        """Initialize a deformable encoder layer."""
        super().__init__()
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.act = act
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, pos: torch.Tensor | None) -> torch.Tensor:
        """Add positional embeddings to the input tensor, if provided."""
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src: torch.Tensor) -> torch.Tensor:
        """Run the feed-forward network."""
        src2 = self.linear2(self.dropout2(self.act(self.linear1(src))))
        src = src + self.dropout3(src2)
        return self.norm2(src)

    def forward(
        self,
        src: torch.Tensor,
        shapes: list,
        reference_points: torch.Tensor,
        pos: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run deformable self-attention followed by FFN."""
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, shapes, padding_mask)
        src = self.norm1(src + self.dropout1(src2))
        return self.forward_ffn(src)


class DeformableTransformerEncoder(nn.Module):
    """Multiscale deformable transformer encoder for BCHW feature map lists."""

    def __init__(
        self,
        ch: list[int],
        hidden_dim: int = 256,
        num_layers: int = 6,
        n_heads: int = 8,
        d_ffn: int = 1024,
        dropout: float = 0.0,
        n_points: int = 4,
        act: nn.Module = nn.ReLU(),
    ):
        """Initialize the encoder with per-level input projections and deformable encoder layers."""
        super().__init__()
        if not isinstance(ch, (list, tuple)):
            raise TypeError(f"DeformableTransformerEncoder expects a list of input channels, but got {type(ch)}.")
        self.hidden_dim = hidden_dim
        self.n_levels = len(ch)
        self.input_proj = nn.ModuleList(
            nn.Sequential(nn.Conv2d(c, hidden_dim, 1, bias=False), nn.BatchNorm2d(hidden_dim)) for c in ch
        )
        encoder_layer = DeformableTransformerEncoderLayer(
            hidden_dim, n_heads, d_ffn, dropout, act, self.n_levels, n_points
        )
        self.layers = _get_clones(encoder_layer, num_layers)

    @staticmethod
    def _get_reference_points(shapes: list[list[int]], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Build normalized per-level grid reference points with shape (1, total_tokens, n_levels, 2)."""
        reference_points = []
        for h, w in shapes:
            ref_y, ref_x = (
                torch.meshgrid(
                    torch.linspace(0.5, h - 0.5, h, dtype=dtype, device=device),
                    torch.linspace(0.5, w - 0.5, w, dtype=dtype, device=device),
                    indexing="ij",
                )
                if TORCH_1_11
                else torch.meshgrid(
                    torch.linspace(0.5, h - 0.5, h, dtype=dtype, device=device),
                    torch.linspace(0.5, w - 0.5, w, dtype=dtype, device=device),
                )
            )
            reference_points.append(torch.stack((ref_x.reshape(-1) / w, ref_y.reshape(-1) / h), -1))
        return torch.cat(reference_points, 0)[None, :, None, :].expand(-1, -1, len(shapes), -1)

    def _flatten_features(self, x: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, list[list[int]]]:
        """Project feature maps and flatten them into source and position token sequences."""
        srcs, pos_embeds, shapes = [], [], []
        for proj, feat in zip(self.input_proj, x):
            src = proj(feat)
            h, w = src.shape[-2:]
            shapes.append([h, w])
            srcs.append(src.flatten(2).permute(0, 2, 1))
            pos = AIFI.build_2d_sincos_position_embedding(w, h, self.hidden_dim)
            pos_embeds.append(pos.to(device=src.device, dtype=src.dtype))
        return torch.cat(srcs, 1), torch.cat(pos_embeds, 1), shapes

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        """Encode a list of multiscale feature maps and return feature maps with unchanged spatial shapes."""
        if not isinstance(x, list):
            raise TypeError(f"DeformableTransformerEncoder expects a list of feature maps, but got {type(x)}.")
        if len(x) != self.n_levels:
            raise ValueError(f"Expected {self.n_levels} feature levels, but got {len(x)}.")

        src, pos, shapes = self._flatten_features(x)
        reference_points = self._get_reference_points(shapes, src.dtype, src.device).expand(src.shape[0], -1, -1, -1)
        for layer in self.layers:
            src = layer(src, shapes, reference_points, pos)

        outputs, start = [], 0
        for h, w in shapes:
            tokens = h * w
            output = src[:, start : start + tokens].transpose(1, 2).reshape(src.shape[0], self.hidden_dim, h, w)
            outputs.append(output.contiguous())
            start += tokens
        return outputs


class DeformableTransformerDecoderLayer(nn.Module):
    """Deformable Transformer Decoder Layer inspired by PaddleDetection and Deformable-DETR implementations.

    This class implements a single decoder layer with self-attention, cross-attention using multiscale deformable
    attention, and a feedforward network.

    Attributes:
        self_attn (nn.MultiheadAttention): Self-attention module.
        dropout1 (nn.Dropout): Dropout after self-attention.
        norm1 (nn.LayerNorm): Layer normalization after self-attention.
        cross_attn (MSDeformAttn): Cross-attention module.
        dropout2 (nn.Dropout): Dropout after cross-attention.
        norm2 (nn.LayerNorm): Layer normalization after cross-attention.
        linear1 (nn.Linear): First linear layer in the feedforward network.
        act (nn.Module): Activation function.
        dropout3 (nn.Dropout): Dropout in the feedforward network.
        linear2 (nn.Linear): Second linear layer in the feedforward network.
        dropout4 (nn.Dropout): Dropout after the feedforward network.
        norm3 (nn.LayerNorm): Layer normalization after the feedforward network.

    References:
        https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
        https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/deformable_transformer.py
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_ffn: int = 1024,
        dropout: float = 0.0,
        act: nn.Module | None = None,
        n_levels: int = 4,
        n_points: int = 4,
    ):
        """Initialize the DeformableTransformerDecoderLayer with the given parameters.

        Args:
            d_model (int): Model dimension.
            n_heads (int): Number of attention heads.
            d_ffn (int): Dimension of the feedforward network.
            dropout (float): Dropout probability.
            act (nn.Module): Activation function.
            n_levels (int): Number of feature levels.
            n_points (int): Number of sampling points.
        """
        super().__init__()

        # Self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.act = nn.ReLU() if act is None else act
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, pos: torch.Tensor | None) -> torch.Tensor:
        """Add positional embeddings to the input tensor, if provided."""
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt: torch.Tensor) -> torch.Tensor:
        """Perform forward pass through the Feed-Forward Network part of the layer.

        Args:
            tgt (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after FFN.
        """
        tgt2 = self.linear2(self.dropout3(self.act(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        return self.norm3(tgt)

    def forward(
        self,
        embed: torch.Tensor,
        refer_bbox: torch.Tensor,
        feats: torch.Tensor,
        shapes: list,
        padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Perform the forward pass through the entire decoder layer.

        Args:
            embed (torch.Tensor): Input embeddings.
            refer_bbox (torch.Tensor): Reference bounding boxes.
            feats (torch.Tensor): Feature maps.
            shapes (list): Feature shapes.
            padding_mask (torch.Tensor, optional): Padding mask.
            attn_mask (torch.Tensor, optional): Attention mask.
            query_pos (torch.Tensor, optional): Query position embeddings.

        Returns:
            (torch.Tensor): Output tensor after decoder layer.
        """
        # Self attention
        q = k = self.with_pos_embed(embed, query_pos)
        tgt = self.self_attn(q, k, embed, attn_mask=attn_mask, need_weights=False)[0]
        embed = embed + self.dropout1(tgt)
        embed = self.norm1(embed)

        # Cross attention
        tgt = self.cross_attn(
            self.with_pos_embed(embed, query_pos), refer_bbox.unsqueeze(2), feats, shapes, padding_mask
        )
        embed = embed + self.dropout2(tgt)
        embed = self.norm2(embed)

        # FFN
        return self.forward_ffn(embed)


class DeformableTransformerDecoder(nn.Module):
    """Deformable Transformer Decoder based on PaddleDetection implementation.

    This class implements a complete deformable transformer decoder with multiple decoder layers and prediction heads
    for bounding box regression and classification.

    Attributes:
        layers (nn.ModuleList): List of decoder layers.
        num_layers (int): Number of decoder layers.
        hidden_dim (int): Hidden dimension.
        eval_idx (int): Index of the layer to use during evaluation.

    References:
        https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    """

    def __init__(self, hidden_dim: int, decoder_layer: nn.Module, num_layers: int, eval_idx: int = -1):
        """Initialize the DeformableTransformerDecoder with the given parameters.

        Args:
            hidden_dim (int): Hidden dimension.
            decoder_layer (nn.Module): Decoder layer module.
            num_layers (int): Number of decoder layers.
            eval_idx (int): Index of the layer to use during evaluation.
        """
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(
        self,
        embed: torch.Tensor,  # decoder embeddings
        refer_bbox: torch.Tensor,  # anchor
        feats: torch.Tensor,  # image features
        shapes: list,  # feature shapes
        bbox_head: nn.Module,
        score_head: nn.Module,
        pos_mlp: nn.Module,
        attn_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ):
        """Perform the forward pass through the entire decoder.

        Args:
            embed (torch.Tensor): Decoder embeddings.
            refer_bbox (torch.Tensor): Reference bounding boxes.
            feats (torch.Tensor): Image features.
            shapes (list): Feature shapes.
            bbox_head (nn.Module): Bounding box prediction head.
            score_head (nn.Module): Score prediction head.
            pos_mlp (nn.Module): Position MLP.
            attn_mask (torch.Tensor, optional): Attention mask.
            padding_mask (torch.Tensor, optional): Padding mask.

        Returns:
            dec_bboxes (torch.Tensor): Decoded bounding boxes.
            dec_cls (torch.Tensor): Decoded classification scores.
        """
        output = embed
        dec_bboxes = []
        dec_cls = []
        # Match inverse_sigmoid(sigmoid(x), eps=1e-5) behavior while avoiding repeated inverse_sigmoid calls.
        eps = 1e-5
        logit_min = math.log(eps / (1.0 - eps))
        logit_max = -logit_min
        refer_bbox_logits_det = refer_bbox.clamp(min=logit_min, max=logit_max)
        refer_bbox = refer_bbox_logits_det.sigmoid()
        refer_bbox_logits_for_loss = refer_bbox_logits_det
        for i, layer in enumerate(self.layers):
            output = layer(output, refer_bbox, feats, shapes, padding_mask, attn_mask, pos_mlp(refer_bbox))

            bbox = bbox_head[i](output)
            ref_logits = refer_bbox_logits_for_loss if (self.training and i > 0) else refer_bbox_logits_det
            refined_logits = bbox + ref_logits
            refined_bbox = refined_logits.sigmoid()

            if self.training:
                dec_cls.append(score_head[i](output))
                dec_bboxes.append(refined_bbox)
            elif i == self.eval_idx:
                dec_cls.append(score_head[i](output))
                dec_bboxes.append(refined_bbox)
                break

            refer_bbox_logits_for_loss = refined_logits.clamp(min=logit_min, max=logit_max)
            refer_bbox_logits_det = refer_bbox_logits_for_loss.detach() if self.training else refer_bbox_logits_for_loss
            refer_bbox = refer_bbox_logits_det.sigmoid()

        return torch.stack(dec_bboxes), torch.stack(dec_cls)


class RoPEViT(nn.Module):
    def __init__(
        self,
        c1: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        ffn_ratio: float = 4.0,
        rope_rescale_coords: float | None = None,
    ):
        super().__init__()
        self.c1, self.embed_dim, self.num_heads, self.num_layers = c1, embed_dim, num_heads, num_layers

        self.proj = nn.Linear(c1, embed_dim) if c1 != embed_dim else nn.Identity()
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dtype=torch.float32,
            rescale_coords=rope_rescale_coords,
        )
        self.blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=ffn_ratio,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    init_values=1e-5,
                    mask_k_bias=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self._init_dinov3_style_weights()

    def _init_dinov3_style_weights(self):
        """Align initialization with DINOv3 ViT blocks (Linear/LayerNorm/LayerScale and masked K-bias)."""
        self.rope_embed._init_weights()
        for module in self.modules():
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                if hasattr(module, "bias_mask") and module.bias_mask is not None:
                    o = module.out_features
                    module.bias_mask.fill_(1)
                    module.bias_mask[o // 3 : 2 * o // 3].fill_(0)
            elif isinstance(module, nn.LayerNorm):
                module.reset_parameters()
            elif isinstance(module, LayerScale):
                module.reset_parameters()

    def prepare_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        if x.ndim != 4:
            raise ValueError(f"RoPEViT expects a BCHW tensor, but got shape {tuple(x.shape)}")
        _, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x, (h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, (h, w) = self.prepare_tokens(x)
        rope_sincos = self.rope_embed(H=h, W=w)
        for blk in self.blocks:
            x = blk(x, rope_sincos)
        x = self.norm(x)
        return x.transpose(1, 2).reshape(x.shape[0], self.embed_dim, h, w).contiguous()
