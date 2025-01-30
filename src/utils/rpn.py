import torch
import torch.nn as nn
import torchvision

def sample_positive_neagative(labels, positive_count, total_count):
    positive = torch.where(labels >= 1)[0]
    negative = torch.where(labels == 0)[0]
    num_pos = positive_count
    num_pos = min(positive.numel(), num_pos)
    num_neg = total_count - num_pos
    num_neg = min(negative.numel(), num_neg)
    perm_positive_idxs = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
    perm_negative_idxs = torch.randperm(negative.numel(), device=negative.device)[:num_neg]
    
    pos_idxs = positive[perm_positive_idxs]
    neg_idxs = negative[perm_negative_idxs]
    sampled_pos_idx_mask = torch.zeros_like(labels, dtype=torch.bool)
    sampled_neg_idx_mask = torch.zeros_like(labels, dtype=torch.bool)
    sampled_pos_idx_mask[pos_idxs] = True
    sampled_neg_idx_mask[neg_idxs] = True
    return sampled_neg_idx_mask, sampled_pos_idx_mask
    

def boxes_to_transformation_targets(gt_boxes, anchors_or_proposals):
    # Get xc, yc, w, h from x1, y1, x2, y2 for anchors
    widths = anchors_or_proposals[:, 2] - anchors_or_proposals[:, 0]
    heights = anchors_or_proposals[:, 3] - anchors_or_proposals[:, 1]
    xc = anchors_or_proposals[:, 0] + 0.5*widths
    yc = anchors_or_proposals[:, 1] + 0.5*heights
    
    # Get xc, yc, w, h from x1, y1, x2, y2 for gt boxes
    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1]
    gt_xc = gt_boxes[:, 0] + 0.5*gt_widths
    gt_yc = gt_boxes[:, 1] + 0.5*gt_heights
    
    target_dx = (gt_xc - xc) / widths
    target_dy = (gt_yc - yc) / heights
    target_xc = torch.log(gt_widths / widths)
    target_yc = torch.log(gt_heights / heights)
    
    target = torch.stack((target_dx, target_dy, target_xc, target_yc),dim=1)
    
    return target
    

def get_iou(boxes1, boxes2):
    """
    Args:
        boxes1: (N*4)
        boxes2: (M*4)
    Return:
        IOU matrix of shape (MxN)
    """
    
    # Area of boxes (x2-x1) * (y2-y1)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Get top left x1, y1
    x_left = torch.max(boxes1[:, None, 0], boxes2[:, 0])     # (N, M)
    y_top = torch.max(boxes1[:, None, 1], boxes2[:, 1])     # (N, M)
    
    # Get bottom right x2, y2
    x_right = torch.min(boxes1[:, None, 3], boxes2[:, 3])
    y_bottom = torch.min(boxes1[:, None, 4], boxes2[:, 4])
    
    intersection_area = torch.max(0, x_right - x_left) * torch.min(0, y_bottom - y_top)
    union = area1[:, None] + area2 - intersection_area
    
    return intersection_area / union    # (N, M)

def apply_regression_pred_to_anchor_or_proposals(box_transform_pred, anchors_or_proposals):
    """
    Args:
        box_transform_pred: (num_anchors_or_proposals, num_classes, 4)
        anchors_or_proposals: (num_anchors_or_proposals, 4)

    Returns:
        pred_boxes: (num_anchors_or_proposals, num_classes, 4)
    """
    
    box_transform_pred = box_transform_pred.reshape(box_transform_pred.size(0), -1, 4)
    
    # Get xc, yc, w, h from x1, y1, x2, y2
    w = anchors_or_proposals[:, 2] - anchors_or_proposals[:, 0]
    h = anchors_or_proposals[:, 3] - anchors_or_proposals[:, 1]
    
    xc = anchors_or_proposals[:, 0] + 0.5 * w
    yc = anchors_or_proposals[:, 1] + 0.5 * h
    
    # dh -> (num_anchors_or_proposals, num_classes)
    dx = box_transform_pred[..., 0]
    dy = box_transform_pred[..., 1]
    dw = box_transform_pred[..., 2]
    dh = box_transform_pred[..., 3]
    
    # pred_xc -> (num_anchors_or_proposals, num_classes)
    pred_xc = dx * w[:, None] + xc[:, None]
    pred_yc = dy * h[:, None] + yc[:, None]
    pred_w = torch.exp(dw) + w[:, None]
    pred_h = torch.exp(dh) + h[:, None]

    # convert into xmin, ymin, xmax, ymax
    pred_box_x1 = pred_xc - 0.5 * pred_w
    pred_box_y1 = pred_yc - 0.5 * pred_h
    pred_box_x2 = pred_xc + 0.5 * pred_w
    pred_box_y2 = pred_yc + 0.5 * pred_h
    
    # pred_boxes -> (num_anchors_or_proposals, num_classes, 4)
    pred_boxes = torch.stack((pred_box_x1, pred_box_y1, pred_box_x2, pred_box_y2), dim=2)
    
    return pred_boxes


def clamp_boxes_to_image_boundary(boxes, image_shape):
    boxes_x1 = boxes[..., 0]
    boxes_y1 = boxes[..., 1]
    boxes_x2 = boxes[..., 2]
    boxes_y2 = boxes[..., 3]
    height, width = image_shape[-2:]
    boxes_x1 = boxes_x1.clamp(min=0, max=width)
    boxes_y1 = boxes_y1.clamp(min=0, max=height)
    boxes_x2 = boxes_x2.clamp(min=0, max=width)
    boxes_y2 = boxes_y2.clamp(min=0, max=height)
    
    boxes = torch.cat((boxes_x1, boxes_y1, boxes_x2, boxes_y2), dim=-1)
    return boxes