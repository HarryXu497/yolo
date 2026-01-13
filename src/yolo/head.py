import torch.nn as nn


class YOLOHead(nn.Module):
    def __init__(self):
        super().__init__()

        self._classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * 7 * 7, 4096),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, 7 * 7 * 30)
        )

    def forward(self, x):
        return self._classifier(x)
