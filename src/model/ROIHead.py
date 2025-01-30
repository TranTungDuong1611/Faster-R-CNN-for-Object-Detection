import torch
import torch.nn as nn
import torchvision

from utils.roi_head import *

class ROIHead(nn.Module):
    def __init__(self, num_classes, in_channels=512):
        super(ROIHead, self).__init__()
        self.num_classes = num_classes
        self.pool_size = 7
        self.fc_inner_dim = 1024
        
        self.fc6 = nn.Linear(in_channels * self.pool_size * self.pool_size, self.fc_inner_dim)
        self.fc7 = nn.Linear(self.fc_inner_dim, self.fc_inner_dim)
        self.cls_layer = nn.Linear(self.fc_inner_dim, num_classes)
        self.bbox_reg_layer = nn.Linear(self.fc_inner_dim, self.num_classes * 4)
        
    def assign_target_to_proposal(self, proposals, gt_boxes, gt_labels):
        iou_matrix = get_iou(gt_boxes, proposals)
        best_match_iou, best_match_gt_idx = iou_matrix.max(dim=0)
        below_low_threshold = best_match_gt_idx < 0.5
        
        best_match_gt_idx[below_low_threshold] = -1
        matched_gt_boxes_for_proposals = gt_boxes[best_match_gt_idx.clamp(min=0)]
        
        labels = gt_labels[best_match_gt_idx.clamp(min=0)]
        labels = labels.to(dtype=torch.int64)
        
        background_proposals = best_match_gt_idx == -1
        labels[background_proposals] = 0
        
        return labels, matched_gt_boxes_for_proposals
        
        
    def forward(self, feat, proposals, image_shape, target):
        if self.training and target is not None:
            gt_boxes = target['bboxes'][0]
            gt_labels = target['labels'][0]
            
            # assign labels and gt_boxes for proposals
            labels, matched_gt_boxes_for_proposals = self.assign_target_to_proposal(proposals, gt_boxes, gt_labels)
            
            sampled_neg_idx_mask, sampled_pos_idx_mask = sample_positive_neagative(labels, positive_count=32, total_count=128)
            
            
            