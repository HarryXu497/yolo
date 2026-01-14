from pathlib import Path
from typing import Optional
import torch

from yolo.backbone import YOLOBackbone
from yolo.yolo import YOLO


def get_device() -> str:
    """
    Gets the backend to train on

    :return: a string representing the backend to use
    :rtype: str
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def create_backbone(path: Optional[Path | str] = None) -> YOLOBackbone:
    """
    Creates or loads the YOLO backbone depending on the path

    :param path: Description
    :type path: The path to the weights, or None to initialize the weights randomly.
    :return: The backbone model
    :rtype: YOLOBackbone
    """
    backbone = YOLOBackbone()

    if path:
        backbone.load_state_dict(torch.load(path, weights_only=True))
    else:
        backbone.initialize_weights()

    return backbone


def create_model(
    *,
    backbone_path: Optional[Path | str] = None,
    model_path: Optional[Path | str] = None,
) -> YOLO:
    if isinstance(model_path, str):
        backbone_path = Path(model_path)

    backbone = create_backbone(backbone_path)

    model = YOLO(backbone=backbone)
    model.load_state_dict(model_path)

    return model
