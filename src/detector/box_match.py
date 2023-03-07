import torch
import numpy as np
""" Functions
1. compute_iou [X]
2. compute_intersection [X]
    : iou score계산하는데 필요함
3. encode [X]
4. match_anchor_boxes [X]
"""
def match_anchor_boxes(image_size, anchor_boxes, gt_boxes):
    ignore_index = -1

    positive_anchor_label = 1

    negative_anchor_label = 0

    positive_jaccard_overlap_threshold = 0.8
 
    negative_jaccard_overlap_threshold = 0.5

    # Compute the IoU between anchor and ground truth boxes. # Shape: [M, N]
    
    IoUs = compute_iou(anchor_boxes=torch.unsqueeze(anchor_boxes, dim=1), gt_boxes=gt_boxes)# .squeeze(0)
    # print(IoUs.shape)
    device = gt_boxes.device
    n_gt_boxes = IoUs.size(1) # IoUs.size(1)

    # Declaration and initialisation of a new tensor containing the binary label for each anchor box.
    # For text/non-text classification, a binary label is assigned to each positive (text) or
    # negative (non-text) anchor. It is defined by computing the IoU overlap with the GT bounding box.
    # For now, We do not care about positive/negatives anchors.
    anchor_labels = torch.full(size=(anchor_boxes.shape[0],), fill_value=ignore_index, dtype=torch.int64)
    
    _, best_anchor_for_each_target_index = torch.max(IoUs, dim=0, keepdim=False)
    best_target_for_each_anchor, best_target_for_each_anchor_index = torch.max(IoUs, dim=1)
    # Assigning each GT box to the corresponding maximum-overlap-anchor.
    # print(best_target_for_each_anchor_index.shape, best_anchor_for_each_target_index.shape)
    # B, N = best_anchor_for_each_target_index.shape
    best_target_for_each_anchor_index[best_anchor_for_each_target_index] =torch.arange(n_gt_boxes) # torch.arange(n_gt_boxes, device=device)

    # Ensuring that every GT box has an anchor assigned.
    best_target_for_each_anchor[best_anchor_for_each_target_index] = positive_anchor_label

    # Taking the real labels for each anchor.
    anchor_labels = anchor_labels[best_target_for_each_anchor_index]

    # A positive anchor is defined as : 
    # an anchor that has an > IoU overlap threshold with any GT box;
    anchor_labels[best_target_for_each_anchor > positive_jaccard_overlap_threshold] = positive_anchor_label

    # The negative anchors are defined as < IoU overlap threshold with all GT boxes.
    anchor_labels[best_target_for_each_anchor < negative_jaccard_overlap_threshold] = negative_anchor_label

    # Finally, we ignore anchor boxes that are outside the image.
    img_h, img_w = image_size
    
    outside_anchors = torch.where(
        (anchor_boxes[:, 0] < 0) |
        (anchor_boxes[:, 1] < 0) |
        (anchor_boxes[:, 2] > img_w) |
        (anchor_boxes[:, 3] > img_h)
    )[0]
    anchor_labels[outside_anchors] = ignore_index

    # calculate bounding box targets.
    gt_boxes = gt_boxes.squeeze(0)
    matched_gt_bboxes = gt_boxes[best_target_for_each_anchor_index]
    bbox_targets = encode(matched_gt_bboxes, anchor_boxes)

    output = (bbox_targets, anchor_labels)

    return output

def compute_iou(anchor_boxes, gt_boxes, eps = 1e-6):
    intersection = compute_intersection(anchor_boxes, gt_boxes)
    anchor_box_areas = (anchor_boxes[..., 2] - anchor_boxes[..., 0] + 1.) * \
        (anchor_boxes[..., 3] - anchor_boxes[..., 1] + 1.) 
    gt_box_areas = (gt_boxes[..., 2] - gt_boxes[..., 0] + 1.) * \
        (gt_boxes[..., 3] - gt_boxes[..., 1] + 1.)
    
    union_area = anchor_box_areas + gt_box_areas - intersection
    return intersection / (union_area + eps)

def compute_intersection(anchor_boxes, gt_boxes):
    """
    - anchor_boxes는 앞서 구한 text detector 모델이 예측하는 score의 기반이 되는 anchor box를 의미한다.
    - gt_boxes는 실제 정답 bounding box로 알고 있는 값을 의미한다.
    - anchor_boxes: (M, 1, 4) gt_boxes: (1, N, 4)
    - returns: (M, N)
    """
    overlaps_top_left = torch.maximum(
        anchor_boxes[..., :2], gt_boxes[..., :2]
    )
    overlap_bottom_right = torch.minimum(
        anchor_boxes[..., 2:], gt_boxes[..., 2:]
    )
    diff = overlap_bottom_right - overlaps_top_left
    ## 제일 큰 x좌표와 y좌표의 차이가 0보다 작을수도 있기 때문에 거의 ReLU 함수를 시도하는 것이라고 봐도 무방하다.
    max_ = torch.maximum(diff, torch.as_tensor(0.0, device = gt_boxes.device))
    intersection = max_[..., 0] * max_[...,1]

    return intersection

def encode(gt_boxes: torch.Tensor, anchor_boxes: torch.Tensor):
    """ bounding box의 위치를 기반으로 y축의 중심점과 세로 길이를 계산한다. (어차피 가로 길이는 고정)
    gt_boxes: 정답 bounding box
    anchor_boxes: 직접 만든 가로 길이가 16으로 일정한 anchor box
    """
    h = gt_boxes[:, 3] - gt_boxes[:, 1] + 1. ## gtbox의 높이
    ha = anchor_boxes[:, 3] - anchor_boxes[:, 1] + 1. ## anchor box의 높이
    Cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2. ## gtbox의 y축 중심
    Cya = (anchor_boxes[:, 1] + anchor_boxes[:, 3]) / 2. ## anchor box의 y축 중심

    Vc = (Cy - Cya) / ha 
    Vh = torch.log(h / ha)
    bbox = torch.stack([Vc, Vh], dim=1)

    return bbox