import torch
import torch.nn as nn

from yolo.backbone import YOLOBackbone


class YOLOPretrain(nn.Module):
    """
    Wraps the YOLO backbone and adds on additional layers to allow for training.
    """
    def __init__(self, backbone: YOLOBackbone, num_outputs=1000):
        super().__init__()
        self._backbone = backbone
        self._avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self._classifier = nn.Linear(in_features=1024, out_features=num_outputs)

    def forward(self, x: torch.Tensor):
        """
        Applies the layer to the input x.
        
        :param x: The input tensor.
        """
        x = self._backbone(x)
        x = self._avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        return self._classifier(x)

    @property
    def backbone(self):
        """
        :returns: The YOLO backbone
        :rtype: YOLOBackbone
        """
        return self._backbone
