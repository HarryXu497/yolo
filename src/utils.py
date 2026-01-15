from pathlib import Path
from typing import Optional
import torch

from yolo.backbone import YOLOBackbone
from yolo.pretrain import YOLOPretrain
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


def create_backbone(*, backbone_path: Optional[Path | str] = None) -> YOLOBackbone:
    """
    Creates or loads the YOLO backbone depending on the path.

    :param path: The path to the weights, or None to initialize the weights randomly.
    :type path: Optional[Path | str]
    :return: The backbone model.
    :rtype: YOLOBackbone
    """
    backbone = YOLOBackbone()

    if backbone_path:
        backbone.load_state_dict(torch.load(backbone_path, weights_only=True))
    else:
        backbone.initialize_weights()

    return backbone


def create_pretrain(*, pretrain_path: Optional[Path | str] = None, num_outputs=100) -> YOLOPretrain:
    """
    Creates or loads the pretraining model depending on the path.

    :param pretrain_path: The path to the weights, or None to initialize the weights randomly.
    :type pretrain_path: Optional[Path | str]
    :return: The pretraining model
    :rtype: YOLOBackbone
    """
    backbone = create_backbone()
    model = YOLOPretrain(backbone=backbone, num_outputs=num_outputs)

    if pretrain_path is not None:
        model.load_state_dict(torch.load(pretrain_path, weights_only=True))

    return model


def create_model_from_backbone_only(
    *,
    backbone_path: Optional[Path | str] = None,
) -> YOLO:
    """
    Creates a YOLO model whose backbone weights can be populated from `backbone_path`.
    
    :param backbone_path: The path to the backbone weights, or None to initialize from scratch. 
    :type backbone_path: Optional[Path | str]
    :return: The YOLO model.
    :rtype: YOLO
    """
    backbone = create_backbone(backbone_path=backbone_path)
    return YOLO(backbone=backbone)


def create_model_from_all_weights(
    *,
    model_path: Optional[Path | str] = None,
) -> YOLO:
    backbone = create_backbone()
    
    model = YOLO(backbone=backbone)

    if model_path is not None:
        model.load_state_dict(torch.load(model_path, weights_only=True))

    return model