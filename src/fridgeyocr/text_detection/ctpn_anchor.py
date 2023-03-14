"""ctpn_anchor.py
1. generate_all_anchor_boxes ## 다양한 세로 길이에 맞게 anchor box를 만들어 줌
2. generate_basis_anchors ## 고정된 16이라는 가로 길이를 기반으로 다양한 세로 길이를 갖는 anchor box를 만든다.
3. scale_anchor ## 실제 이미지 크기 안에 들어가게 range check
4. AnchorMatcher ## 실제 학습을 할때에 필요로 하는 object이다.(일정한 pos sample, neg sample 비율을 맞춰줌)
    |_ jaccard_index
    |_ compute_intersection
    |_ match_anchor_boxes
5. encode
"""
import torch
import numpy as np
import os, sys
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from box_match import match_anchor_boxes

from typing import List, Tuple, Optional

def scale_anchor(shape, basis_anchor):
    """ scales the anchor based on the widths ans heights
    shape: anchor box의 크기 정보 (apply_along_axis 함수를 사용해서 각각의 axis=1마다 함수 적용을 한다.)
    """
    H, W = shape
    center_X = (basis_anchor[0] + basis_anchor[2]) / 2.
    center_Y = (basis_anchor[1] + basis_anchor[3]) / 2.
    scaled_anchor = basis_anchor.copy()

    scaled_anchor[0] = center_X - W / 2. ## X min
    scaled_anchor[1] = center_Y - H / 2. ## Y min
    scaled_anchor[2] = center_X + W / 2. ## X max
    scaled_anchor[3] = center_Y + H / 2. ## Y max

    return scaled_anchor

def generate_basic_anchors(anchor_heights, anchor_shift):
    """ 
    - CTPN 학습을 위해서 만들어야 하는 기준이 되는 anchor detecting box는 가로의 길이가 anchor_shift와 같고, 
    height는 anchor_height에 있는 길이 대로이다.
    """
    basic_anchor = np.array([0, 0, anchor_shift - 1, anchor_shift - 1], np.float32)
    heights = np.array(anchor_heights, dtype=np.float32)
    widths = np.ones(len(heights), dtype = np.float32) * anchor_shift
    sizes = np.column_stack((heights, widths))
    basic_anchors = np.apply_along_axis(
        func1d = scale_anchor, axis = 1, arr = sizes,basis_anchor= basic_anchor
    ) ## 위에서 정의한 scale_anchor 함수를 사용하여서 
    return basic_anchors

"""generate all anchor box
1. 실제 CNN 모델을 거친 output feature map의 크기가 (H / 16, W / 16)이 된다.
2. 그 CNN의 output feature map을 기반으로 만들어질 수 있는 모든 anchor을 all_anchor에 담아 보냄
"""
def generate_all_anchor_boxes(feature_map_size, feature_stride, anchor_heights, anchor_shift):
    basis_anchors = generate_basic_anchors(anchor_heights, anchor_shift)
    anchor_n = basis_anchors.shape[0]
    feat_map_h, feat_map_w = feature_map_size
    all_anchors = np.zeros(
        shape = (anchor_n * feat_map_h * feat_map_w, 4),
        dtype = np.float32
    ) ## 전체 anchor의 개수가 anchor_n * feat_map_h * feat_map_w이 되기 때문에 각각에 순서대로 anchor의
    # (minX, minY, maxX, maxY)를 저장해 준다.
    index = 0 
    for y in range(feat_map_h):
        for x in range(feat_map_w):
            shift = np.array([x, y, x, y]) * feature_stride
            all_anchors[index:index+anchor_n, :] = basis_anchors + shift
            index += anchor_n
    return all_anchors


class TargetTransform(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.name = 'Transform'

    def __call__(self, gt_boxes, image_size, return_anchor_boxes):
        H, W = image_size
        anchor_shift = 16

        feature_map_size = [int(np.ceil(H / anchor_shift)), int(np.ceil(W / anchor_shift))]
        anchor_boxes = generate_all_anchor_boxes(
            feature_map_size = feature_map_size,
            feat_stride = self.cfg.FEATURE_STRIDE,
            anchor_heights = self.cfg.ANCHOR_HEIGHTS,
            anchor_shift = anchor_shift
        )
        matches = match_anchor_boxes(
            cfg = self.cfg,
            image_size = image_size, 
            anchor_boxes = torch.as_tensor(anchor_boxes, device = torch.device("cpu")),
            gt_boxes = gt_boxes
        )

        if return_anchor_boxes:
            return matches + (anchor_boxes,)
        return matches

