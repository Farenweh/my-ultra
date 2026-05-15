from __future__ import annotations

import pytest
import torch

from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.models.utils.ops import get_cdn_group


def _assert_loss_dict_close(
    lhs: dict[str, torch.Tensor],
    rhs: dict[str, torch.Tensor],
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> None:
    assert set(lhs.keys()) == set(rhs.keys())
    for k in sorted(lhs.keys()):
        lv, rv = lhs[k], rhs[k]
        assert torch.isfinite(lv).all(), f"{k} has non-finite value in lhs: {lv}"
        assert torch.isfinite(rv).all(), f"{k} has non-finite value in rhs: {rv}"
        assert torch.allclose(lv, rv, atol=atol, rtol=rtol), f"{k} mismatch: {lv} vs {rv}"


def _build_case(*, with_gt: bool, with_dn: bool):
    torch.manual_seed(42)
    bs = 2
    nc = 20
    hidden_dim = 32
    num_queries = 64
    num_dn = 30
    decoder_layers = 3

    if with_gt:
        gt_groups = [3, 2]
        batch_idx = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
        cls = torch.randint(0, nc, (sum(gt_groups),), dtype=torch.long)
        bboxes = torch.rand(sum(gt_groups), 4)
    else:
        gt_groups = [0, 0]
        batch_idx = torch.zeros(0, dtype=torch.long)
        cls = torch.zeros(0, dtype=torch.long)
        bboxes = torch.zeros(0, 4)

    batch = {
        "cls": cls,
        "bboxes": bboxes,
        "batch_idx": batch_idx,
        "gt_groups": gt_groups,
    }

    main_layers = decoder_layers + 1  # encoder + decoder
    pred_bboxes = torch.rand(main_layers, bs, num_queries, 4)
    pred_scores = torch.randn(main_layers, bs, num_queries, nc)

    dn_bboxes = None
    dn_scores = None
    dn_meta = None
    if with_gt and with_dn:
        class_embed = torch.randn(nc, hidden_dim)
        _, _, _, dn_meta = get_cdn_group(
            batch=batch,
            num_classes=nc,
            num_queries=num_queries,
            class_embed=class_embed,
            num_dn=num_dn,
            training=True,
        )
        assert dn_meta is not None
        dn_q = int(dn_meta["dn_num_split"][0])
        dn_bboxes = torch.rand(decoder_layers, bs, dn_q, 4)
        dn_scores = torch.randn(decoder_layers, bs, dn_q, nc)

    return {
        "batch": batch,
        "pred_bboxes": pred_bboxes,
        "pred_scores": pred_scores,
        "dn_bboxes": dn_bboxes,
        "dn_scores": dn_scores,
        "dn_meta": dn_meta,
        "nc": nc,
    }


@pytest.mark.parametrize("use_uni_match", [False, True])
@pytest.mark.parametrize("with_dn", [False, True])
def test_rtdetr_loss_batched_matches_legacy(use_uni_match: bool, with_dn: bool):
    case = _build_case(with_gt=True, with_dn=with_dn)
    criterion = RTDETRDetectionLoss(nc=case["nc"], use_vfl=True, use_uni_match=use_uni_match)
    criterion.device = case["pred_bboxes"].device

    layer_match_indices = criterion._collect_match_indices_serial(
        case["pred_bboxes"],
        case["pred_scores"],
        case["batch"]["bboxes"],
        case["batch"]["cls"],
        case["batch"]["gt_groups"],
    )
    loss_batched_main = criterion._forward_group_batched(
        case["pred_bboxes"], case["pred_scores"], case["batch"], postfix="", layer_match_indices=layer_match_indices
    )
    loss_legacy_main = criterion._forward_group_legacy(
        case["pred_bboxes"], case["pred_scores"], case["batch"], postfix="", layer_match_indices=layer_match_indices
    )
    _assert_loss_dict_close(loss_batched_main, loss_legacy_main)

    if with_dn:
        dn_match_indices = criterion.get_dn_match_indices(
            case["dn_meta"]["dn_pos_idx"], case["dn_meta"]["dn_num_group"], case["batch"]["gt_groups"]
        )
        layer_match_indices_dn = [dn_match_indices for _ in range(int(case["dn_bboxes"].shape[0]))]
        loss_batched_dn = criterion._forward_group_batched(
            case["dn_bboxes"],
            case["dn_scores"],
            case["batch"],
            postfix="_dn",
            layer_match_indices=layer_match_indices_dn,
        )
        loss_legacy_dn = criterion._forward_group_legacy(
            case["dn_bboxes"],
            case["dn_scores"],
            case["batch"],
            postfix="_dn",
            layer_match_indices=layer_match_indices_dn,
        )
        _assert_loss_dict_close(loss_batched_dn, loss_legacy_dn)

    criterion._use_legacy_loss = False
    loss_new = criterion(
        preds=(case["pred_bboxes"], case["pred_scores"]),
        batch=case["batch"],
        dn_bboxes=case["dn_bboxes"],
        dn_scores=case["dn_scores"],
        dn_meta=case["dn_meta"],
    )
    criterion._use_legacy_loss = True
    loss_old = criterion(
        preds=(case["pred_bboxes"], case["pred_scores"]),
        batch=case["batch"],
        dn_bboxes=case["dn_bboxes"],
        dn_scores=case["dn_scores"],
        dn_meta=case["dn_meta"],
    )
    _assert_loss_dict_close(loss_new, loss_old)


def test_rtdetr_loss_batched_matches_legacy_no_gt():
    case = _build_case(with_gt=False, with_dn=False)
    criterion = RTDETRDetectionLoss(nc=case["nc"], use_vfl=True, use_uni_match=False)
    criterion.device = case["pred_bboxes"].device

    layer_match_indices = criterion._collect_match_indices_serial(
        case["pred_bboxes"],
        case["pred_scores"],
        case["batch"]["bboxes"],
        case["batch"]["cls"],
        case["batch"]["gt_groups"],
    )
    loss_batched_main = criterion._forward_group_batched(
        case["pred_bboxes"], case["pred_scores"], case["batch"], postfix="", layer_match_indices=layer_match_indices
    )
    loss_legacy_main = criterion._forward_group_legacy(
        case["pred_bboxes"], case["pred_scores"], case["batch"], postfix="", layer_match_indices=layer_match_indices
    )
    _assert_loss_dict_close(loss_batched_main, loss_legacy_main)
