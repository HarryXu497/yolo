import torch


def compute_iou(box_1: torch.Tensor, box_2: torch.Tensor) -> torch.Tensor:
    """
    Computes the pairwise Intersection over Union (IoU) for two batches of boxes.

    :param box_1: The first batch of bounding boxes. Should be of dimension (batch, 49, 4) in format (x, y, w, h)
    :type box_1: torch.Tensor
    :param box_2: The second batch of bounding boxes. Should be of dimension (batch, 49, 4) in format (x, y, w, h)
    :type box_2: torch.Tensor

    :return: A tensor of dimension (batch, 49) of the IoU of each pair of boxes.
    :rtype: torch.Tensor
    """
    # Compute the coordinates of the corners for each box
    box_1_x1, box_1_y1 = box_1[..., 0] - \
        box_1[..., 2]/2, box_1[..., 1] - box_1[..., 3]/2
    box_1_x2, box_1_y2 = box_1[..., 0] + \
        box_1[..., 2]/2, box_1[..., 1] + box_1[..., 3]/2

    box_2_x1, box_2_y1 = box_2[..., 0] - box_2[..., 2] / \
        2, box_2[..., 1] - box_2[..., 3]/2
    box_2_x2, box_2_y2 = box_2[..., 0] + box_2[..., 2] / \
        2, box_2[..., 1] + box_2[..., 3]/2

    # Compute intersection coordinates
    inter_x1 = torch.max(box_1_x1, box_2_x1)
    inter_y1 = torch.max(box_1_y1, box_2_y1)
    inter_x2 = torch.min(box_1_x2, box_2_x2)
    inter_y2 = torch.min(box_1_y2, box_2_y2)

    intersection = (inter_x2 - inter_x1).clamp(0) * \
        (inter_y2 - inter_y1).clamp(0)
    union = box_1[..., 2] * box_1[..., 3] + \
        box_2[..., 2] * box_2[..., 3] - intersection

    # Add epsilon to avoid division by 0
    return intersection / (union + 1e-6)
