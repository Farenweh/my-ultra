from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.modules.transformer import DeformableTransformerDecoder, DeformableTransformerDecoderLayer, MLP


def test_deformable_decoder_component_forward_and_backward():
    torch.manual_seed(0)
    bs = 2
    num_queries = 32
    hidden_dim = 64
    num_heads = 8
    num_levels = 3
    num_layers = 2
    num_classes = 20
    shapes = [[8, 8], [4, 4], [2, 2]]
    total_tokens = sum(h * w for h, w in shapes)

    layer = DeformableTransformerDecoderLayer(
        d_model=hidden_dim,
        n_heads=num_heads,
        d_ffn=hidden_dim * 4,
        dropout=0.0,
        n_levels=num_levels,
        n_points=4,
    )
    decoder = DeformableTransformerDecoder(hidden_dim=hidden_dim, decoder_layer=layer, num_layers=num_layers)
    bbox_head = nn.ModuleList([MLP(hidden_dim, hidden_dim, 4, num_layers=3) for _ in range(num_layers)])
    score_head = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for _ in range(num_layers)])
    pos_mlp = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

    embed = torch.randn(bs, num_queries, hidden_dim, requires_grad=True)
    refer_bbox = torch.rand(bs, num_queries, 4)
    feats = torch.randn(bs, total_tokens, hidden_dim, requires_grad=True)
    attn_mask = torch.zeros(num_queries, num_queries, dtype=torch.bool)

    dec_bboxes, dec_scores = decoder(
        embed=embed,
        refer_bbox=refer_bbox,
        feats=feats,
        shapes=shapes,
        bbox_head=bbox_head,
        score_head=score_head,
        pos_mlp=pos_mlp,
        attn_mask=attn_mask,
    )

    assert dec_bboxes.shape == (num_layers, bs, num_queries, 4)
    assert dec_scores.shape == (num_layers, bs, num_queries, num_classes)
    assert torch.isfinite(dec_bboxes).all()
    assert torch.isfinite(dec_scores).all()

    loss = dec_bboxes.sum() + dec_scores.sum()
    loss.backward()
    assert embed.grad is not None and torch.isfinite(embed.grad).all()
    assert feats.grad is not None and torch.isfinite(feats.grad).all()
