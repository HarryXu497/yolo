import torch


IMG_WIDTH = 448
IMG_HEIGHT = 448


def convert_bounding_box(bbox: torch.Tensor, grid_x: int, grid_y: int, S=7):
    x, y, w, h, _ = bbox

    x_relative = (x + grid_x) / S
    y_relative = (y + grid_y) / S
    x_center = x_relative * IMG_WIDTH
    y_center = y_relative * IMG_HEIGHT

    box_width = w * IMG_WIDTH
    box_height = h * IMG_HEIGHT

    x1 = x_center - box_width / 2
    y1 = y_center - box_height / 2
    x2 = x_center + box_width / 2
    y2 = y_center + box_height / 2

    return x1.item(), y1.item(), x2.item(), y2.item()


def iou(bbox_pred, bbox_actual):
    x1_predicted, y1_predicted, x2_predicted, y2_predicted = convert_bounding_box(
        bbox_pred)
    x1_actual, y1_actual, x2_actual, y2_actual = convert_bounding_box(
        bbox_actual)

    x_intersection_left = max(x1_predicted, x1_actual)
    x_intersection_right = min(x2_predicted, x2_actual)
    y_intersection_upper = max(y1_predicted, y1_actual)
    y_intersection_lower = min(y2_predicted, y2_actual)

    w_inter = max(0, x_intersection_right - x_intersection_left)
    h_inter = max(0, y_intersection_lower - y_intersection_upper)

    w_box_predicted = max(0, x2_predicted - x1_predicted)
    h_box_predicted = max(0, y2_predicted - y1_predicted)

    w_box_actual = max(0, x2_actual - x1_actual)
    h_box_actual = max(0, y2_actual - y1_actual)

    area_predicted = w_box_predicted * h_box_predicted
    area_actual = w_box_actual * h_box_actual

    area_intersection = w_inter * h_inter

    area_union = area_predicted + area_actual - area_intersection

    if area_union == 0:
        return 0
    else:
        return area_intersection / area_union
