# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ultralytics.utils.loss import FocalLoss, VarifocalLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.checks import IS_ASCEND

from .ops import HungarianMatcher


class DETRLoss(nn.Module):
    """DETR (DEtection TRansformer) Loss class for calculating various loss components.

    This class computes classification loss, bounding box loss, GIoU loss, and optionally auxiliary losses for the DETR
    object detection model.

    Attributes:
        nc (int): Number of classes.
        loss_gain (dict[str, float]): Coefficients for different loss components.
        aux_loss (bool): Whether to compute auxiliary losses.
        use_fl (bool): Whether to use FocalLoss.
        use_vfl (bool): Whether to use VarifocalLoss.
        use_uni_match (bool): Whether to use a fixed layer for auxiliary branch label assignment.
        uni_match_ind (int): Index of fixed layer to use if use_uni_match is True.
        matcher (HungarianMatcher): Object to compute matching cost and indices.
        fl (FocalLoss | None): Focal Loss object if use_fl is True, otherwise None.
        vfl (VarifocalLoss | None): Varifocal Loss object if use_vfl is True, otherwise None.
        device (torch.device): Device on which tensors are stored.
    """

    def __init__(
        self,
        nc: int = 80,
        loss_gain: dict[str, float] | None = None,
        aux_loss: bool = True,
        use_fl: bool = True,
        use_vfl: bool = False,
        use_uni_match: bool = False,
        uni_match_ind: int = 0,
        gamma: float = 1.5,
        alpha: float = 0.25,
    ):
        """Initialize DETR loss function with customizable components and gains.

        Uses default loss_gain if not provided. Initializes HungarianMatcher with preset cost gains. Supports auxiliary
        losses and various loss types.

        Args:
            nc (int): Number of classes.
            loss_gain (dict[str, float], optional): Coefficients for different loss components.
            aux_loss (bool): Whether to use auxiliary losses from each decoder layer.
            use_fl (bool): Whether to use FocalLoss.
            use_vfl (bool): Whether to use VarifocalLoss.
            use_uni_match (bool): Whether to use fixed layer for auxiliary branch label assignment.
            uni_match_ind (int): Index of fixed layer for uni_match.
            gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
            alpha (float): The balancing factor used to address class imbalance.
        """
        super().__init__()

        if loss_gain is None:
            loss_gain = {"class": 1, "bbox": 5, "giou": 2, "no_object": 0.1, "mask": 1, "dice": 1}
        self.nc = nc
        self.matcher = HungarianMatcher(cost_gain={"class": 2, "bbox": 5, "giou": 2})
        self.loss_gain = loss_gain
        self.aux_loss = aux_loss
        self.fl = FocalLoss(gamma, alpha) if use_fl else None
        self.vfl = VarifocalLoss(gamma, alpha) if use_vfl else None

        self.use_uni_match = use_uni_match
        self.uni_match_ind = uni_match_ind
        self.device = None

    def _get_loss_class(
        self, pred_scores: torch.Tensor, targets: torch.Tensor, gt_scores: torch.Tensor, num_gts: int, postfix: str = ""
    ) -> dict[str, torch.Tensor]:
        """Compute classification loss based on predictions, target values, and ground truth scores.

        Args:
            pred_scores (torch.Tensor): Predicted class scores with shape (B, N, C).
            targets (torch.Tensor): Target class indices with shape (B, N).
            gt_scores (torch.Tensor): Ground truth confidence scores with shape (B, N).
            num_gts (int): Number of ground truth objects.
            postfix (str, optional): String to append to the loss name for identification in multi-loss scenarios.

        Returns:
            (dict[str, torch.Tensor]): Dictionary containing classification loss value.

        Notes:
            The function supports different classification loss types:
            - Varifocal Loss (if self.vfl is not None and num_gts > 0)
            - Focal Loss (if self.fl is not None)
            - BCE Loss (default fallback)
        """
        # Logits: [b, query, num_classes], gt_class: list[[n, 1]]
        name_class = f"loss_class{postfix}"
        bs, nq = pred_scores.shape[:2]
        # one_hot = F.one_hot(targets, self.nc + 1)[..., :-1]  # (bs, num_queries, num_classes)
        one_hot = torch.zeros((bs, nq, self.nc + 1), dtype=torch.int64, device=targets.device)
        one_hot.scatter_(2, targets.unsqueeze(-1), 1)
        one_hot = one_hot[..., :-1]
        gt_scores = gt_scores.view(bs, nq, 1) * one_hot

        if self.fl:
            if num_gts and self.vfl:
                loss_cls = self.vfl(pred_scores, gt_scores, one_hot)
            else:
                loss_cls = self.fl(pred_scores, one_hot.float())
            loss_cls /= max(num_gts, 1) / nq
        else:
            loss_cls = nn.BCEWithLogitsLoss(reduction="none")(pred_scores, gt_scores).mean(1).sum()  # YOLO CLS loss

        return {name_class: loss_cls.squeeze() * self.loss_gain["class"]}

    def _get_loss_bbox(
        self, pred_bboxes: torch.Tensor, gt_bboxes: torch.Tensor, postfix: str = ""
    ) -> dict[str, torch.Tensor]:
        """Compute bounding box and GIoU losses for predicted and ground truth bounding boxes.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes with shape (N, 4).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes with shape (N, 4).
            postfix (str, optional): String to append to the loss names for identification in multi-loss scenarios.

        Returns:
            (dict[str, torch.Tensor]): Dictionary containing:
                - loss_bbox{postfix}: L1 loss between predicted and ground truth boxes, scaled by the bbox loss gain.
                - loss_giou{postfix}: GIoU loss between predicted and ground truth boxes, scaled by the giou loss gain.

        Notes:
            If no ground truth boxes are provided (empty list), zero-valued tensors are returned for both losses.
        """
        # Boxes: [b, query, 4], gt_bbox: list[[n, 4]]
        name_bbox = f"loss_bbox{postfix}"
        name_giou = f"loss_giou{postfix}"

        loss = {}
        if len(gt_bboxes) == 0:
            loss[name_bbox] = torch.tensor(0.0, device=self.device)
            loss[name_giou] = torch.tensor(0.0, device=self.device)
            return loss

        loss[name_bbox] = self.loss_gain["bbox"] * F.l1_loss(pred_bboxes, gt_bboxes, reduction="sum") / len(gt_bboxes)
        loss[name_giou] = 1.0 - bbox_iou(pred_bboxes, gt_bboxes, xywh=True, GIoU=True)
        loss[name_giou] = loss[name_giou].sum() / len(gt_bboxes)
        loss[name_giou] = self.loss_gain["giou"] * loss[name_giou]
        return {k: v.squeeze() for k, v in loss.items()}

    # This function is for future RT-DETR Segment models
    # def _get_loss_mask(self, masks, gt_mask, match_indices, postfix=''):
    #     # masks: [b, query, h, w], gt_mask: list[[n, H, W]]
    #     name_mask = f'loss_mask{postfix}'
    #     name_dice = f'loss_dice{postfix}'
    #
    #     loss = {}
    #     if sum(len(a) for a in gt_mask) == 0:
    #         loss[name_mask] = torch.tensor(0., device=self.device)
    #         loss[name_dice] = torch.tensor(0., device=self.device)
    #         return loss
    #
    #     num_gts = len(gt_mask)
    #     src_masks, target_masks = self._get_assigned_bboxes(masks, gt_mask, match_indices)
    #     src_masks = F.interpolate(src_masks.unsqueeze(0), size=target_masks.shape[-2:], mode='bilinear')[0]
    #     # TODO: torch does not have `sigmoid_focal_loss`, but it's not urgent since we don't use mask branch for now.
    #     loss[name_mask] = self.loss_gain['mask'] * F.sigmoid_focal_loss(src_masks, target_masks,
    #                                                                     torch.tensor([num_gts], dtype=torch.float32))
    #     loss[name_dice] = self.loss_gain['dice'] * self._dice_loss(src_masks, target_masks, num_gts)
    #     return loss

    # This function is for future RT-DETR Segment models
    # @staticmethod
    # def _dice_loss(inputs, targets, num_gts):
    #     inputs = F.sigmoid(inputs).flatten(1)
    #     targets = targets.flatten(1)
    #     numerator = 2 * (inputs * targets).sum(1)
    #     denominator = inputs.sum(-1) + targets.sum(-1)
    #     loss = 1 - (numerator + 1) / (denominator + 1)
    #     return loss.sum() / num_gts

    def _get_loss_aux(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_cls: torch.Tensor,
        gt_groups: list[int],
        match_indices: list[tuple] | None = None,
        postfix: str = "",
        masks: torch.Tensor | None = None,
        gt_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Get auxiliary losses for intermediate decoder layers.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes from auxiliary layers.
            pred_scores (torch.Tensor): Predicted scores from auxiliary layers.
            gt_bboxes (torch.Tensor): Ground truth bounding boxes.
            gt_cls (torch.Tensor): Ground truth classes.
            gt_groups (list[int]): Number of ground truths per image.
            match_indices (list[tuple], optional): Pre-computed matching indices.
            postfix (str, optional): String to append to loss names.
            masks (torch.Tensor, optional): Predicted masks if using segmentation.
            gt_mask (torch.Tensor, optional): Ground truth masks if using segmentation.

        Returns:
            (dict[str, torch.Tensor]): Dictionary of auxiliary losses.
        """
        # NOTE: loss class, bbox, giou, mask, dice
        loss = torch.zeros(5 if masks is not None else 3, device=pred_bboxes.device)
        if match_indices is None and self.use_uni_match:
            match_indices = self.matcher(
                pred_bboxes[self.uni_match_ind],
                pred_scores[self.uni_match_ind],
                gt_bboxes,
                gt_cls,
                gt_groups,
                masks=masks[self.uni_match_ind] if masks is not None else None,
                gt_mask=gt_mask,
            )
        for i, (aux_bboxes, aux_scores) in enumerate(zip(pred_bboxes, pred_scores)):
            aux_masks = masks[i] if masks is not None else None
            loss_ = self._get_loss(
                aux_bboxes,
                aux_scores,
                gt_bboxes,
                gt_cls,
                gt_groups,
                masks=aux_masks,
                gt_mask=gt_mask,
                postfix=postfix,
                match_indices=match_indices,
            )
            loss[0] += loss_[f"loss_class{postfix}"]
            loss[1] += loss_[f"loss_bbox{postfix}"]
            loss[2] += loss_[f"loss_giou{postfix}"]
            # if masks is not None and gt_mask is not None:
            #     loss_ = self._get_loss_mask(aux_masks, gt_mask, match_indices, postfix)
            #     loss[3] += loss_[f'loss_mask{postfix}']
            #     loss[4] += loss_[f'loss_dice{postfix}']

        loss = {
            f"loss_class_aux{postfix}": loss[0],
            f"loss_bbox_aux{postfix}": loss[1],
            f"loss_giou_aux{postfix}": loss[2],
        }
        # if masks is not None and gt_mask is not None:
        #     loss[f'loss_mask_aux{postfix}'] = loss[3]
        #     loss[f'loss_dice_aux{postfix}'] = loss[4]
        return loss

    def _collect_match_indices_serial(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_cls: torch.Tensor,
        gt_groups: list[int],
        match_indices: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        masks: torch.Tensor | None = None,
        gt_mask: torch.Tensor | None = None,
    ) -> list[list[tuple[torch.Tensor, torch.Tensor]]]:
        """Collect per-layer Hungarian match indices with serial solving."""
        num_layers = int(pred_bboxes.shape[0])
        if num_layers == 0:
            return []

        if match_indices is not None:
            return [match_indices for _ in range(num_layers)]

        layer_match_indices: list[list[tuple[torch.Tensor, torch.Tensor]] | None] = [None] * num_layers
        aux_layers = num_layers - 1

        if aux_layers > 0 and self.aux_loss:
            if self.use_uni_match:
                uni_layer = max(0, min(self.uni_match_ind, aux_layers - 1))
                uni_masks = masks[uni_layer] if masks is not None else None
                shared_indices = self.matcher(
                    pred_bboxes[uni_layer],
                    pred_scores[uni_layer],
                    gt_bboxes,
                    gt_cls,
                    gt_groups,
                    masks=uni_masks,
                    gt_mask=gt_mask,
                )
                for i in range(aux_layers):
                    layer_match_indices[i] = shared_indices
            else:
                for i in range(aux_layers):
                    aux_masks = masks[i] if masks is not None else None
                    layer_match_indices[i] = self.matcher(
                        pred_bboxes[i], pred_scores[i], gt_bboxes, gt_cls, gt_groups, masks=aux_masks, gt_mask=gt_mask
                    )

        last_masks = masks[-1] if masks is not None else None
        layer_match_indices[-1] = self.matcher(
            pred_bboxes[-1], pred_scores[-1], gt_bboxes, gt_cls, gt_groups, masks=last_masks, gt_mask=gt_mask
        )
        return [m for m in layer_match_indices if m is not None]

    def _pack_layer_indices(
        self, layer_match_indices: list[list[tuple[torch.Tensor, torch.Tensor]]]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Pack per-layer match indices into [L, M] tensors for batched gather."""
        if not layer_match_indices:
            return None

        batch_idx_layers: list[torch.Tensor] = []
        src_idx_layers: list[torch.Tensor] = []
        gt_idx_layers: list[torch.Tensor] = []
        match_size: int | None = None
        index_device: torch.device | None = None

        for match_indices in layer_match_indices:
            (batch_idx, src_idx), gt_idx = self._get_index(match_indices)
            current_size = int(batch_idx.numel())
            if match_size is None:
                match_size = current_size
                index_device = batch_idx.device
            elif current_size != match_size:
                return None

            if index_device is not None:
                batch_idx = batch_idx.to(index_device)
                src_idx = src_idx.to(index_device)
                gt_idx = gt_idx.to(index_device)
            batch_idx_layers.append(batch_idx)
            src_idx_layers.append(src_idx)
            gt_idx_layers.append(gt_idx)

        return (
            torch.stack(batch_idx_layers, dim=0),
            torch.stack(src_idx_layers, dim=0),
            torch.stack(gt_idx_layers, dim=0),
        )

    def _forward_group_legacy(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        batch: dict[str, Any],
        postfix: str = "",
        match_indices: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        layer_match_indices: list[list[tuple[torch.Tensor, torch.Tensor]]] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Legacy per-layer loss implementation used for reference and equivalence checks."""
        gt_cls, gt_bboxes, gt_groups = batch["cls"], batch["bboxes"], batch["gt_groups"]

        if layer_match_indices is None:
            total_loss = self._get_loss(
                pred_bboxes[-1],
                pred_scores[-1],
                gt_bboxes,
                gt_cls,
                gt_groups,
                postfix=postfix,
                match_indices=match_indices,
            )
            if self.aux_loss:
                total_loss.update(
                    self._get_loss_aux(
                        pred_bboxes[:-1], pred_scores[:-1], gt_bboxes, gt_cls, gt_groups, match_indices, postfix
                    )
                )
            return total_loss

        num_layers = int(pred_bboxes.shape[0])
        if len(layer_match_indices) != num_layers:
            return self._forward_group_legacy(
                pred_bboxes, pred_scores, batch, postfix=postfix, match_indices=match_indices
            )

        total_loss = self._get_loss(
            pred_bboxes[-1],
            pred_scores[-1],
            gt_bboxes,
            gt_cls,
            gt_groups,
            postfix=postfix,
            match_indices=layer_match_indices[-1],
        )
        if self.aux_loss:
            aux = pred_scores.new_zeros(3)
            for i in range(max(num_layers - 1, 0)):
                loss_i = self._get_loss(
                    pred_bboxes[i],
                    pred_scores[i],
                    gt_bboxes,
                    gt_cls,
                    gt_groups,
                    postfix=postfix,
                    match_indices=layer_match_indices[i],
                )
                aux[0] += loss_i[f"loss_class{postfix}"]
                aux[1] += loss_i[f"loss_bbox{postfix}"]
                aux[2] += loss_i[f"loss_giou{postfix}"]
            total_loss.update(
                {
                    f"loss_class_aux{postfix}": aux[0],
                    f"loss_bbox_aux{postfix}": aux[1],
                    f"loss_giou_aux{postfix}": aux[2],
                }
            )
        return total_loss

    def _forward_group_batched(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        batch: dict[str, Any],
        postfix: str,
        layer_match_indices: list[list[tuple[torch.Tensor, torch.Tensor]]],
    ) -> dict[str, torch.Tensor]:
        """Batched tensor loss path with serial Hungarian indices."""
        ascend_npu = IS_ASCEND and pred_bboxes.device.type == "npu"
        if len(layer_match_indices) != int(pred_bboxes.shape[0]):
            if ascend_npu:
                raise RuntimeError(
                    "Ascend NPU RT-DETR batched loss 要求每个预测层都有一组匹配索引；"
                    f"当前 {len(layer_match_indices)} 组索引对应 {int(pred_bboxes.shape[0])} 个预测层。"
                )
            return self._forward_group_legacy(
                pred_bboxes, pred_scores, batch, postfix=postfix, layer_match_indices=layer_match_indices
            )

        packed_indices = self._pack_layer_indices(layer_match_indices)
        if packed_indices is None:
            if ascend_npu:
                raise RuntimeError(
                    "Ascend NPU RT-DETR batched loss 要求各预测层匹配数量一致；拒绝通过 legacy 路径隐藏优化失败。"
                )
            return self._forward_group_legacy(
                pred_bboxes, pred_scores, batch, postfix=postfix, layer_match_indices=layer_match_indices
            )

        gt_cls, gt_bboxes = batch["cls"], batch["bboxes"]
        num_layers, bs, nq = pred_scores.shape[:3]
        batch_idx, src_idx, gt_idx = packed_indices
        match_count = int(batch_idx.shape[1])

        if batch_idx.device != pred_bboxes.device:
            batch_idx = batch_idx.to(pred_bboxes.device)
            src_idx = src_idx.to(pred_bboxes.device)
        gt_idx_box = gt_idx.to(gt_bboxes.device)
        gt_idx_cls = gt_idx.to(gt_cls.device)

        layer_idx = (
            torch.arange(num_layers, device=pred_bboxes.device, dtype=torch.long).unsqueeze(1).expand(-1, match_count)
        )

        if match_count > 0:
            pred_match = pred_bboxes[layer_idx, batch_idx, src_idx]
            gt_match = gt_bboxes[gt_idx_box].to(pred_match.device)
            iou_match = bbox_iou(pred_match.detach(), gt_match, xywh=True).squeeze(-1)
        else:
            pred_match = pred_bboxes.new_zeros((num_layers, 0, 4))
            gt_match = pred_match
            iou_match = pred_scores.new_zeros((num_layers, 0))

        if self.fl and match_count > 0 and self.vfl:
            pred_float = pred_scores.float()
            pred_prob = pred_float.sigmoid()

            # VFL normalizes by the number of matched targets, so use the all-negative loss as a base term and
            # correct only the matched positive class locations. This avoids building dense one-hot targets.
            neg_loss = F.softplus(pred_float) * (self.vfl.alpha * pred_prob.pow(self.vfl.gamma))
            loss_cls_layer = neg_loss.sum((1, 2, 3))

            target_cls = gt_cls[gt_idx_cls]
            if target_cls.device != pred_scores.device:
                target_cls = target_cls.to(pred_scores.device)
            pred_pos = pred_float[layer_idx, batch_idx, src_idx, target_cls]
            pos_score = iou_match.to(pred_float.dtype)
            neg_pos_loss = F.softplus(pred_pos) * (self.vfl.alpha * pred_pos.sigmoid().pow(self.vfl.gamma))
            pos_loss = (F.softplus(pred_pos) - pred_pos * pos_score) * pos_score
            loss_cls_layer = (loss_cls_layer + (pos_loss - neg_pos_loss).sum(1)) / match_count
        else:
            targets = torch.full((num_layers, bs, nq), self.nc, dtype=gt_cls.dtype, device=pred_scores.device)
            if match_count > 0:
                targets[layer_idx, batch_idx, src_idx] = gt_cls[gt_idx_cls]

            gt_score_map = pred_scores.new_zeros((num_layers, bs, nq))
            if match_count > 0:
                gt_score_map[layer_idx, batch_idx, src_idx] = iou_match.to(gt_score_map.dtype)

            one_hot = torch.zeros((num_layers, bs, nq, self.nc + 1), dtype=torch.int64, device=pred_scores.device)
            one_hot.scatter_(3, targets.unsqueeze(-1), 1)
            one_hot = one_hot[..., :-1]
            gt_scores = gt_score_map.unsqueeze(-1) * one_hot

            if self.fl:
                label_float = one_hot.float()
                pred_prob = pred_scores.sigmoid()
                p_t = label_float * pred_prob + (1 - label_float) * (1 - pred_prob)
                modulating_factor = (1.0 - p_t) ** self.fl.gamma
                loss_cls_layer = F.binary_cross_entropy_with_logits(pred_scores, label_float, reduction="none")
                loss_cls_layer *= modulating_factor
                if (self.fl.alpha > 0).any():
                    alpha = self.fl.alpha.to(device=pred_scores.device, dtype=pred_scores.dtype)
                    alpha_factor = label_float * alpha + (1 - label_float) * (1 - alpha)
                    loss_cls_layer *= alpha_factor
                loss_cls_layer = loss_cls_layer.mean(2).sum((1, 2))
                loss_cls_layer /= max(match_count, 1) / nq
            else:
                loss_cls_layer = (
                    F.binary_cross_entropy_with_logits(pred_scores, gt_scores, reduction="none").mean(2).sum((1, 2))
                )
        loss_cls_layer = loss_cls_layer * self.loss_gain["class"]

        if match_count > 0:
            loss_bbox_layer = (
                self.loss_gain["bbox"] * F.l1_loss(pred_match, gt_match, reduction="none").sum((1, 2)) / match_count
            )
            loss_giou_layer = 1.0 - bbox_iou(pred_match, gt_match, xywh=True, GIoU=True).squeeze(-1)
            loss_giou_layer = self.loss_gain["giou"] * (loss_giou_layer.sum(1) / match_count)
        else:
            loss_bbox_layer = pred_bboxes.new_zeros((num_layers,))
            loss_giou_layer = pred_bboxes.new_zeros((num_layers,))

        total_loss = {
            f"loss_class{postfix}": loss_cls_layer[-1].squeeze(),
            f"loss_bbox{postfix}": loss_bbox_layer[-1].squeeze(),
            f"loss_giou{postfix}": loss_giou_layer[-1].squeeze(),
        }
        if self.aux_loss:
            total_loss.update(
                {
                    f"loss_class_aux{postfix}": loss_cls_layer[:-1].sum()
                    if num_layers > 1
                    else loss_cls_layer.new_tensor(0.0),
                    f"loss_bbox_aux{postfix}": loss_bbox_layer[:-1].sum()
                    if num_layers > 1
                    else loss_bbox_layer.new_tensor(0.0),
                    f"loss_giou_aux{postfix}": loss_giou_layer[:-1].sum()
                    if num_layers > 1
                    else loss_giou_layer.new_tensor(0.0),
                }
            )

        if getattr(self, "_enable_loss_finite_guard", False):
            if not all(torch.isfinite(v).all() for v in total_loss.values()):
                if ascend_npu:
                    raise RuntimeError(
                        "Ascend NPU RT-DETR batched loss 产生非有限值；拒绝通过 legacy 路径隐藏优化失败。"
                    )
                return self._forward_group_legacy(
                    pred_bboxes, pred_scores, batch, postfix=postfix, layer_match_indices=layer_match_indices
                )
        return total_loss

    @staticmethod
    def _get_index(match_indices: list[tuple]) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Extract batch indices, source indices, and destination indices from match indices.

        Args:
            match_indices (list[tuple]): List of tuples containing matched indices.

        Returns:
            batch_idx (tuple[torch.Tensor, torch.Tensor]): Tuple containing (batch_idx, src_idx).
            dst_idx (torch.Tensor): Destination indices.
        """
        non_empty = [(src, dst) for src, dst in match_indices if src.numel() > 0]
        if not non_empty:
            empty = torch.empty(0, dtype=torch.long)
            return (empty, empty), empty

        src_idx = torch.cat([src for src, _ in non_empty])
        dst_idx = torch.cat([dst for _, dst in non_empty])
        lengths = torch.as_tensor([src.numel() for src, _ in match_indices], dtype=torch.long, device=src_idx.device)
        batch_idx = torch.repeat_interleave(torch.arange(len(match_indices), device=src_idx.device), lengths)
        return (batch_idx, src_idx), dst_idx

    def _get_assigned_bboxes(
        self, pred_bboxes: torch.Tensor, gt_bboxes: torch.Tensor, match_indices: list[tuple]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Assign predicted bounding boxes to ground truth bounding boxes based on match indices.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes.
            gt_bboxes (torch.Tensor): Ground truth bounding boxes.
            match_indices (list[tuple]): List of tuples containing matched indices.

        Returns:
            pred_assigned (torch.Tensor): Assigned predicted bounding boxes.
            gt_assigned (torch.Tensor): Assigned ground truth bounding boxes.
        """
        pred_assigned = torch.cat(
            [
                t[i] if len(i) > 0 else torch.zeros(0, t.shape[-1], device=self.device)
                for t, (i, _) in zip(pred_bboxes, match_indices)
            ]
        )
        gt_assigned = torch.cat(
            [
                t[j] if len(j) > 0 else torch.zeros(0, t.shape[-1], device=self.device)
                for t, (_, j) in zip(gt_bboxes, match_indices)
            ]
        )
        return pred_assigned, gt_assigned

    def _get_loss(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_cls: torch.Tensor,
        gt_groups: list[int],
        masks: torch.Tensor | None = None,
        gt_mask: torch.Tensor | None = None,
        postfix: str = "",
        match_indices: list[tuple] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Calculate losses for a single prediction layer.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes.
            pred_scores (torch.Tensor): Predicted class scores.
            gt_bboxes (torch.Tensor): Ground truth bounding boxes.
            gt_cls (torch.Tensor): Ground truth classes.
            gt_groups (list[int]): Number of ground truths per image.
            masks (torch.Tensor, optional): Predicted masks if using segmentation.
            gt_mask (torch.Tensor, optional): Ground truth masks if using segmentation.
            postfix (str, optional): String to append to loss names.
            match_indices (list[tuple], optional): Pre-computed matching indices.

        Returns:
            (dict[str, torch.Tensor]): Dictionary of losses.
        """
        if match_indices is None:
            match_indices = self.matcher(
                pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups, masks=masks, gt_mask=gt_mask
            )

        idx, gt_idx = self._get_index(match_indices)
        pred_bboxes, gt_bboxes = pred_bboxes[idx], gt_bboxes[gt_idx]

        bs, nq = pred_scores.shape[:2]
        targets = torch.full((bs, nq), self.nc, device=pred_scores.device, dtype=gt_cls.dtype)
        targets[idx] = gt_cls[gt_idx]

        gt_scores = torch.zeros([bs, nq], device=pred_scores.device)
        if len(gt_bboxes):
            gt_scores[idx] = bbox_iou(pred_bboxes.detach(), gt_bboxes, xywh=True).squeeze(-1)

        return {
            **self._get_loss_class(pred_scores, targets, gt_scores, len(gt_bboxes), postfix),
            **self._get_loss_bbox(pred_bboxes, gt_bboxes, postfix),
            # **(self._get_loss_mask(masks, gt_mask, match_indices, postfix) if masks is not None and gt_mask is not None else {})
        }

    def forward(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        batch: dict[str, Any],
        postfix: str = "",
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Calculate loss for predicted bounding boxes and scores.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes, shape (L, B, N, 4).
            pred_scores (torch.Tensor): Predicted class scores, shape (L, B, N, C).
            batch (dict[str, Any]): Batch information containing cls, bboxes, and gt_groups.
            postfix (str, optional): Postfix for loss names.
            **kwargs (Any): Additional arguments, may include 'match_indices'.

        Returns:
            (dict[str, torch.Tensor]): Computed losses, including main and auxiliary (if enabled).

        Notes:
            Uses last elements of pred_bboxes and pred_scores for main loss, and the rest for auxiliary losses if
            self.aux_loss is True.
        """
        self.device = pred_bboxes.device
        match_indices = kwargs.get("match_indices", None)
        gt_cls, gt_bboxes, gt_groups = batch["cls"], batch["bboxes"], batch["gt_groups"]
        group_bboxes = pred_bboxes if self.aux_loss else pred_bboxes[-1:]
        group_scores = pred_scores if self.aux_loss else pred_scores[-1:]

        if getattr(self, "_use_legacy_loss", False):
            return self._forward_group_legacy(
                group_bboxes, group_scores, batch, postfix=postfix, match_indices=match_indices
            )

        layer_match_indices = self._collect_match_indices_serial(
            group_bboxes, group_scores, gt_bboxes, gt_cls, gt_groups, match_indices=match_indices
        )
        return self._forward_group_batched(group_bboxes, group_scores, batch, postfix, layer_match_indices)


class RTDETRDetectionLoss(DETRLoss):
    """Real-Time DEtection TRansformer (RT-DETR) Detection Loss class that extends the DETRLoss.

    This class computes the detection loss for the RT-DETR model, which includes the standard detection loss as well as
    an additional denoising training loss when provided with denoising metadata.
    """

    def forward(
        self,
        preds: tuple[torch.Tensor, torch.Tensor],
        batch: dict[str, Any],
        dn_bboxes: torch.Tensor | None = None,
        dn_scores: torch.Tensor | None = None,
        dn_meta: dict[str, Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass to compute detection loss with optional denoising loss.

        Args:
            preds (tuple[torch.Tensor, torch.Tensor]): Tuple containing predicted bounding boxes and scores.
            batch (dict[str, Any]): Batch data containing ground truth information.
            dn_bboxes (torch.Tensor, optional): Denoising bounding boxes.
            dn_scores (torch.Tensor, optional): Denoising scores.
            dn_meta (dict[str, Any], optional): Metadata for denoising.

        Returns:
            (dict[str, torch.Tensor]): Dictionary containing total loss and denoising loss if applicable.
        """
        pred_bboxes, pred_scores = preds
        total_loss = super().forward(pred_bboxes, pred_scores, batch)

        # Check for denoising metadata to compute denoising training loss
        if dn_meta is not None:
            dn_pos_idx, dn_num_group = dn_meta["dn_pos_idx"], dn_meta["dn_num_group"]
            assert len(batch["gt_groups"]) == len(dn_pos_idx)

            # Get the match indices for denoising
            match_indices = self.get_dn_match_indices(dn_pos_idx, dn_num_group, batch["gt_groups"])

            if getattr(self, "_use_legacy_loss", False):
                dn_loss = self._forward_group_legacy(
                    dn_bboxes, dn_scores, batch, postfix="_dn", match_indices=match_indices
                )
            else:
                layer_match_indices = [match_indices for _ in range(int(dn_bboxes.shape[0]))]
                dn_loss = self._forward_group_batched(dn_bboxes, dn_scores, batch, "_dn", layer_match_indices)
            total_loss.update(dn_loss)
        else:
            # If no denoising metadata is provided, set denoising loss to zero
            total_loss.update({f"{k}_dn": torch.tensor(0.0, device=self.device) for k in total_loss})

        return total_loss

    @staticmethod
    def get_dn_match_indices(
        dn_pos_idx: list[torch.Tensor], dn_num_group: int, gt_groups: list[int]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Get match indices for denoising.

        Args:
            dn_pos_idx (list[torch.Tensor]): List of tensors containing positive indices for denoising.
            dn_num_group (int): Number of denoising groups.
            gt_groups (list[int]): List of integers representing number of ground truths per image.

        Returns:
            (list[tuple[torch.Tensor, torch.Tensor]]): List of tuples containing matched indices for denoising.
        """
        dn_match_indices = []
        idx_groups = torch.as_tensor([0, *gt_groups[:-1]]).cumsum_(0)
        for i, num_gt in enumerate(gt_groups):
            if num_gt > 0:
                gt_idx = torch.arange(end=num_gt, dtype=torch.long) + idx_groups[i]
                gt_idx = gt_idx.repeat(dn_num_group)
                assert len(dn_pos_idx[i]) == len(gt_idx), (
                    f"Expected the same length, but got {len(dn_pos_idx[i])} and {len(gt_idx)} respectively."
                )
                dn_match_indices.append((dn_pos_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros([0], dtype=torch.long), torch.zeros([0], dtype=torch.long)))
        return dn_match_indices
