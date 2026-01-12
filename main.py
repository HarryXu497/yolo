import torch
from yolo.backbone import YOLO


model = YOLO()
test_input = torch.randn(1, 3, 448, 448) # Standard YOLOv1 input size
output = model(test_input)
print(output.shape)