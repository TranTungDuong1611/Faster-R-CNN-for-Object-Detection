import numpy as np
import torch
import torch.nn as nn
import torchvision

from utils.utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# RPN module
class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels=512):
        super(RegionProposalNetwork, self).__init__()
        self.in_channels = in_channels
        self.scales = [128, 256, 512]
        self.aspect_ratios = [0.5, 1.0, 2.0]
        self.num_anchors = len(self.scales) * len(self.aspect_ratios)
        
        # 3x3 convolution
        self.rpn_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, stride=1, padding=1)
        
        # 1x1 classification
        self.cls_layer = nn.Conv2d(in_channels=self.in_channels, out_channels=self.num_anchors, kernel_size=1, stride=1, padding=1)
        
        # 1x1 regression
        self.regression_layer = nn.Conv2d(in_channels=self.in_channels, out_channels=self.num_anchors*4, kernel_size=1, stride=1, padding=1)
        
        # Relu activation
        self.relu = nn.ReLU()
        
        
    def filter_proposals(self, proposals, cls_scores, image_shape):
        # Pre NMS Filtering
        cls_scores = cls_scores.reshape(-1)
        cls_scores = torch.sigmoid(cls_scores)
        _, top_n_idx = cls_scores.topk(10000)
        cls_scores = cls_scores[top_n_idx]
        proposals = proposals[top_n_idx]
        
        # clamb box to image boundary
        proposals = clamp_boxes_to_image_boundary(proposals, image_shape)
        
        # NMS base on objectives
        keep_mask = torch.zeros_like(cls_scores, dtype=torch.bool)
        keep_indices = torch.ops.torchvision.nms(proposals, cls_scores, 0.7)
        post_nms_keep_indices = keep_indices[cls_scores[keep_indices].sort(descending=True)[1]]
        
        # Post NMS topk filtering
        proposals = proposals[post_nms_keep_indices[:2000]]
        cls_scores = cls_scores[post_nms_keep_indices[:2000]]
        
        return proposals, cls_scores
        
        
    def generate_anchors(self, image, feat):
        """ function to generate all the anchor boxs in the image

        Args:
            image: (B, C, W, H)
            feat (B, C, W, H):
        
        Return:
            anchor_boxs (list): a list of anchor coordinates coressponds to each location on feature map
        """
        
        grid_h, grid_w = feat.shape[-2:]
        image_h, image_w = image.shape[-2:]
        
        stride_h = torch.tensor(image_h // grid_h, dtype=torch.int64, device=feat.device)
        stride_w = torch.tensor(image_w // grid_w, dtype=torch.int64, device=feat.device)
        
        scales = torch.as_tensor(self.scales, dtype=feat.dtype, device=feat.device)
        aspect_ratios = torch.as_tensor(self.aspect_ratios, dtype=feat.dtype, device=feat.device)
        
        # h/w = aspect_ratio and h*w=1
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios
        
        # Dot product ratio * scales
        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)
                
        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        base_anchors = base_anchors.round()
        
        # Get the shifts in x-axis (0, 1,..., w_feat-1) * stride_w
        shift_x = torch.arange(0, grid_w, dtype=torch.int32, device=feat.device) * stride_w
        
        # Get the shifts in y-axis (0, 1, ..., h_feat-1) * stride_h
        shift_y = torch.arange(0, grid_h, dtype=torch.int32, device=feat.device) * stride_h
        
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
        
        # (H_feat, W_feat)
        shift_x = shift_x.reshape(-1)       # [shift_x, shift_x, ...]
        shift_y = shift_y.reshape(-1)       # [shift_y[0], shift_y[1], ..., shift_y[0], shift_y[1], ....]
        shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)
        
        # shift => (H_feat, W_feat, 4)
        
        # base anchor -> (num_anchors_per_location, 4)
        # shift -> (H_feat, W_feat, 4)
        anchors = (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4))
        # (H_feat + W_feat, num_anchors_per_location, 4)
        
        anchors = anchors.reshape(-1, 4)
        # anchors -> (H_feat + W_feat + num_anchors_per_location, 4)
        
        return anchors
        
    def forward(self, image, feat, target):
        # Call RPN layer
        rpn_feature = self.relu(self.rpn_conv(image))
        cls_scores = self.cls_layer(rpn_feature)
        regression_bbox = self.regression_layer(rpn_feature)
        
        # Generate anchor
        anchors = self.generate_anchors(image, feat)
        
        # cls_score -> (Batch, num_anchor_per_location, H_feat, W_feat)
        num_anchors_per_location = cls_scores.size(1)
        cls_scores = cls_scores.permute(0, 2, 3, 1)
        cls_scores = cls_scores.reshape(-1, 1)
        # cls_scores -> (Batch*num_anchor_per_location*H_fea* W_feat, 1)
        
        # regression_bboxes -> (Batch, num_anchor_per_location * 4, H_feat, W_feat)
        regression_bbox = regression_bbox.view(regression_bbox.size(0), num_anchors_per_location, 4, rpn_feature.shape[-2], rpn_feature.shape[-1])
        regression_bbox = regression_bbox.permute(0, 3, 4, 1, 2)
        regression_bbox = regression_bbox.reshape(-1, 4)
        # regression_bboxes -> (Batch*num_anchor_per_location*H_fea* W_feat, 4)
        
        # Transform generated anchors according to box_transform_pred
        proposals = apply_regression_pred_to_anchor_or_proposals(regression_bbox.detach().reshape(-1, 1, 4), anchors)
        
        proposals = proposals.reshape(proposals.size(0), 4)
        proposals, scores = self.filter_proposals(proposals, cls_scores.detach(), image.shape)
        
        
        
        
# image = torch.rand(3, 100, 100)
# feat = torch.rand(3, 10, 10)

# rpn = RegionProposalNetwork()
# rpn.generate_anchors(image, feat)
