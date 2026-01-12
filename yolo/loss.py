import torch
import torch.nn as nn

from yolo.bounding_box import compute_iou


class YOLOLoss(nn.Module):
    def __init__(self, lambda_coord=5, lambda_noobj=0.5):
        super().__init__()
        self._sum_squared_error_loss = nn.MSELoss(reduction="sum")
        self._mse = nn.MSELoss(reduction="sum")
        self._lambda_coord = lambda_coord
        self._lambda_noobj = lambda_noobj
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss according to the original YOLO paper.

        :param predictions: The predictions tensor output by YOLO. Should be of dimensions (batch, 7, 7, 30)
        :type predictions: torch.Tensor
        :param targets: The target tensors. Should be of dimensions (batch, 7, 7, 25)
        :type targets: torch.Tensor

        :returns: A rank 0 Tensor containing the average loss
        :rtype: torch.Tensor
        """

        # Identify which cells have objects
        exists_box = targets[..., 4:5]

        # Pick the better of the two candidate boxes
        iou_b1 = compute_iou(predictions[..., 0:4], targets[..., 0:4])
        iou_b2 = compute_iou(predictions[..., 5:9], targets[..., 0:4])

        # Dimensions of (batch, 7, 7, 1); 0 for box 1 and 1 for box 2 at each cell
        best_box = torch.argmax(torch.stack(
            [iou_b1, iou_b2], dim=0), dim=0).unsqueeze(-1)

        # Localization loss
        # Gets the bounding box cells for the best box without branches
        pred_box = (1 - best_box) * \
            predictions[..., 0:4] + best_box * predictions[..., 5:9]
        target_box = targets[..., 0:4]

        # Use abs and epsilon to handle negative weights in random initialization
        pred_box[..., 2:4] = torch.sqrt(torch.abs(pred_box[..., 2:4]) + 1e-6)
        target_box[..., 2:4] = torch.sqrt(target_box[..., 2:4])

        loss_coord = self._sum_squared_error_loss(
            exists_box * pred_box,
            exists_box * target_box,
        )

        # Object confidence loss
        # Gets the confidence for the best box without branches
        pred_conf = (1 - best_box) * \
            predictions[..., 4:5] + best_box * predictions[..., 9:10]

        loss_obj = self._sum_squared_error_loss(
            exists_box * pred_conf,
            exists_box * targets[..., 5:6],
        )

        # No object confidence loss
        # Penalize confidence for BOTH boxes if no object is in the cell
        loss_noobj = self._sum_squared_error_loss(
            (1 - exists_box) * predictions[..., 4:5],
            (1 - exists_box) * targets[..., 4:5]
        ) + self._sum_squared_error_loss(
            (1 - exists_box) * predictions[..., 9:10],
            (1 - exists_box) * targets[..., 4:5]
        )

        # Classification loss
        loss_class = self._sum_squared_error_loss(
            exists_box * predictions[..., 10:],
            exists_box * targets[..., 10:]
        )

        # Total loss
        total_loss = (
            self._lambda_coord * loss_coord,
            + loss_obj,
            + self._lambda_noobj * loss_noobj,
            + loss_class,
        )

        # Return average loss over the batch
        return total_loss / predictions.size(0)
