# 定义模型配置字典
dinov2_model_configs = {
    # --- Standard Models (No Registers) ---
    "s": {
        "img_size": 518,
        "patch_size": 14,
        "init_values": 1.0,
        "ffn_layer": "mlp",
        "block_chunks": 0,
        "num_register_tokens": 0,
        "interpolate_antialias": False,
        "interpolate_offset": 0.1,
        # Architecture specific (deduced from common ViT standards / DINOv2 paper for S/14)
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        # Defaults from DinoVisionTransformer that are likely used
        "in_chans": 3,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "ffn_bias": True,
        "proj_bias": True,
        "drop_path_rate": 0.0,
        "drop_path_uniform": False,
        # 'embed_layer': PatchEmbed, # 通常是默认值
        # 'act_layer': nn.GELU,      # 通常是默认值
        # 'block_fn': Block,         # 通常是默认值
    },
    "b": {
        "img_size": 518,
        "patch_size": 14,
        "init_values": 1.0,
        "ffn_layer": "mlp",
        "block_chunks": 0,
        "num_register_tokens": 0,
        "interpolate_antialias": False,
        "interpolate_offset": 0.1,
        # Architecture specific (deduced for B/14)
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        # Defaults from DinoVisionTransformer
        "in_chans": 3,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "ffn_bias": True,
        "proj_bias": True,
        "drop_path_rate": 0.0,
        "drop_path_uniform": False,
    },
    "l": {
        "img_size": 518,
        "patch_size": 14,
        "init_values": 1.0,
        "ffn_layer": "mlp",
        "block_chunks": 0,
        "num_register_tokens": 0,
        "interpolate_antialias": False,
        "interpolate_offset": 0.1,
        # Architecture specific (deduced for L/14)
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        # Defaults from DinoVisionTransformer
        "in_chans": 3,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "ffn_bias": True,
        "proj_bias": True,
        "drop_path_rate": 0.0,
        "drop_path_uniform": False,
    },
    "g": {
        "img_size": 518,
        "patch_size": 14,
        "init_values": 1.0,
        "ffn_layer": "swiglufused",  # Overridden for Giant model
        "block_chunks": 0,
        "num_register_tokens": 0,
        "interpolate_antialias": False,
        "interpolate_offset": 0.1,
        # Architecture specific (deduced for G/14)
        "embed_dim": 1536,
        "depth": 40,
        "num_heads": 24,
        # Defaults from DinoVisionTransformer
        "in_chans": 3,
        "mlp_ratio": 4.0,  # Note: swiglu might implicitly change internal structure ratio
        "qkv_bias": True,
        "ffn_bias": True,
        "proj_bias": True,
        "drop_path_rate": 0.0,
        "drop_path_uniform": False,
    },
    # --- Register Models ---
    "s_reg": {
        "img_size": 518,
        "patch_size": 14,
        "init_values": 1.0,
        "ffn_layer": "mlp",
        "block_chunks": 0,
        "num_register_tokens": 4,  # Register specific
        "interpolate_antialias": True,  # Register specific
        "interpolate_offset": 0.0,  # Register specific
        # Architecture specific (S/14)
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        # Defaults from DinoVisionTransformer
        "in_chans": 3,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "ffn_bias": True,
        "proj_bias": True,
        "drop_path_rate": 0.0,
        "drop_path_uniform": False,
    },
    "b_reg": {
        "img_size": 518,
        "patch_size": 14,
        "init_values": 1.0,
        "ffn_layer": "mlp",
        "block_chunks": 0,
        "num_register_tokens": 4,  # Register specific
        "interpolate_antialias": True,  # Register specific
        "interpolate_offset": 0.0,  # Register specific
        # Architecture specific (B/14)
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        # Defaults from DinoVisionTransformer
        "in_chans": 3,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "ffn_bias": True,
        "proj_bias": True,
        "drop_path_rate": 0.0,
        "drop_path_uniform": False,
    },
    "l_reg": {
        "img_size": 518,
        "patch_size": 14,
        "init_values": 1.0,
        "ffn_layer": "mlp",
        "block_chunks": 0,
        "num_register_tokens": 4,  # Register specific
        "interpolate_antialias": True,  # Register specific
        "interpolate_offset": 0.0,  # Register specific
        # Architecture specific (L/14)
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        # Defaults from DinoVisionTransformer
        "in_chans": 3,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "ffn_bias": True,
        "proj_bias": True,
        "drop_path_rate": 0.0,
        "drop_path_uniform": False,
    },
    "g_reg": {
        "img_size": 518,
        "patch_size": 14,
        "init_values": 1.0,
        "ffn_layer": "swiglufused",  # Overridden for Giant model
        "block_chunks": 0,
        "num_register_tokens": 4,  # Register specific
        "interpolate_antialias": True,  # Register specific
        "interpolate_offset": 0.0,  # Register specific
        # Architecture specific (G/14)
        "embed_dim": 1536,
        "depth": 40,
        "num_heads": 24,
        # Defaults from DinoVisionTransformer
        "in_chans": 3,
        "mlp_ratio": 4.0,  # Note: swiglu might implicitly change internal structure ratio
        "qkv_bias": True,
        "ffn_bias": True,
        "proj_bias": True,
        "drop_path_rate": 0.0,
        "drop_path_uniform": False,
    },
}
dinov2_pretrained_urls = {
    "s": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
    "b": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
    "l": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
    "g": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth",
    "s_reg": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth",
    "b_reg": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth",
    "l_reg": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth",
    "g_reg": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_reg4_pretrain.pth",
}
