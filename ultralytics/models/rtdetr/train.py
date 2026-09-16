# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy

import torch

from ultralytics.data.utils import get_split_fraction
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import RANK
from ultralytics.utils.torch_utils import unwrap_model

from .val import RTDETRValidator, build_rtdetr_dataset


class RTDETRTrainer(DetectionTrainer):
    """Trainer class for the RT-DETR model developed by Baidu for real-time object detection.

    This class extends the DetectionTrainer class for YOLO to adapt to the specific features and architecture of
    RT-DETR. The model leverages Vision Transformers and has capabilities like IoU-aware query selection and adaptable
    inference speed.

    Attributes:
        loss_names (tuple): Names of the loss components, derived from the loss dict returned by the criterion.
        data (dict): Dataset configuration containing class count and other parameters.
        args (dict): Training arguments and hyperparameters.
        save_dir (Path): Directory to save training results.
        test_loader (DataLoader): DataLoader for validation/testing data.

    Methods:
        get_model: Initialize and return an RT-DETR model for object detection tasks.
        build_dataset: Build and return an RT-DETR dataset for training or validation.
        get_validator: Return an RTDETRValidator suitable for RT-DETR model validation.

    Examples:
        >>> from ultralytics.models.rtdetr.train import RTDETRTrainer
        >>> args = dict(model="rtdetr-l.yaml", data="coco8.yaml", imgsz=640, epochs=3)
        >>> trainer = RTDETRTrainer(overrides=args)
        >>> trainer.train()

    Notes:
        - F.grid_sample used in RT-DETR does not support the `deterministic=True` argument.
        - AMP training can lead to NaN outputs and may produce errors during bipartite graph matching.
    """

    def get_model(self, cfg: str | dict | None = None, weights: torch.nn.Module | None = None, verbose: bool = True):
        """Initialize and return an RT-DETR model for object detection tasks.

        Args:
            cfg (str | dict, optional): Model configuration file path or dictionary.
            weights (torch.nn.Module, optional): Pretrained model whose weights are loaded into the new model.
            verbose (bool): Verbose logging if True.

        Returns:
            (RTDETRDetectionModel): Initialized model.
        """
        model = self.set_model_names_for_load(
            RTDETRDetectionModel(
                cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1, summary=False
            )
        )
        if weights:
            model.load(weights)
        return model

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Build and return an RT-DETR dataset for training or validation.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): Dataset mode, either 'train' or 'val'.
            batch (int, optional): Batch size for rectangle training.

        Returns:
            (RTDETRDataset): Dataset object for the specific mode.
        """
        return build_rtdetr_dataset(self.args, img_path, batch, self.data, mode)

    def _model_forward(self, batch):
        """编译前准备 GT，并通过原有模型包装执行含去噪 query 的前向。"""
        if not self.args.compile:
            return super()._model_forward(batch)
        model = unwrap_model(self.model)
        targets = model._prepare_targets(batch)
        preds = self.model(batch["img"], batch=targets)
        return model.loss(batch, preds, targets=targets)

    def _get_ddp_static_graph(self) -> bool:
        """GT 数量和去噪分支随 batch 变化，不启用静态 DDP 图。"""
        return False

    def _get_ddp_loss_scale(self) -> int:
        """RT-DETR 已按匹配目标数归一化，保持各 rank 损失的梯度平均。"""
        return 1

    def get_validator(self):
        """Return an RTDETRValidator suitable for RT-DETR model validation."""
        return RTDETRValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
