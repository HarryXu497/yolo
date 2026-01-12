import torch
import torch.nn as nn

from yolo.bounding_box import compute_iou
from yolo.head import YOLOHead


class YOLO(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self._conv_layers = nn.Sequential(
            backbone,

            # 5, continued. Add in the last 4 conv layers
            nn.Conv2d(in_channels=1024, out_channels=1024,
                      kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=1024, out_channels=1024,
                      kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(negative_slope=0.1),

            # 6
            nn.Conv2d(in_channels=1024, out_channels=1024,
                      kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(in_channels=1024, out_channels=1024,
                      kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
        )

        self._head = YOLOHead()

    def forward(self, x):
        output = self._conv_layers(x)
        output = self._head(output)
        output = torch.reshape(output, (-1, 7, 7, 30))
        return output
