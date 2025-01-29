import numpy as np
import torch
import torch.nn as nn
import torchvision

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def apply_regression_pred_to_anchor_or_proposals(box_transform_pred, anchors_or_proposals):
    """_summary_

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
    
    pred_boxes = torch.stack((pred_box_x1, pred_box_y1, pred_box_x2, pred_box_y2), dim=2)

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
        
    def generate_anchors(self, image, feat):
        """ function to generate all the anchor boxs in the image

        Args:
            image ():
            feat ():
        
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
        
        
        
        
        
        
# image = torch.rand(3, 100, 100)
# feat = torch.rand(3, 10, 10)

# rpn = RegionProposalNetwork()
# rpn.generate_anchors(image, feat)

arr1 = torch.tensor([1, 2, 3, 4])
arr2 = torch.tensor([2, 3, 4, 4])
arr3 = torch.tensor([1, 2, 3, 4])
print(arr1.shape)
stack = torch.stack((arr1[:, None], arr2[:, None], arr3[:, None]), dim=2)
print(stack.shape)