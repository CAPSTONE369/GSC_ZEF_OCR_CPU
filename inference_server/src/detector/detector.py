import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import os, sys
from collections import defaultdict
from loguru import logger
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ctpn import CTPN
STD = [0.20037157, 0.18366718, 0.19631825]
MEAN = [0.90890862, 0.91631571, 0.90724233]
BASE = '/content/drive/MyDrive/SpeakingFridgey/model_weights/detection'

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE, 'src', 'utils'))
from connector import TextProposalConnector
from anchor import generate_all_anchor_boxes

""" (1) clip_boxes
- bounding box를 H, W, 0.0, 0.0에 대해서 내부 길이로 정해질 수 있도록 한다.
"""
def clip_boxes(bboxes, image_size):
    ## 원본 이미지의 크기의 가로, 세로보다 넘치거나 0보다 작은 길이를 가질수도 있어서 그부분 예외 처리
    H, W = image_size
    zero = 0.0
    W_diff, H_diff = W - 1., H - 1.
    bboxes[:, 0::2] = np.maximum(np.minimum(bboxes[:, 0::2], W_diff), zero)
    bboxes[:, 1::2] = np.maximum(np.minimum(bboxes[:, 1::2], H_diff), zero)

    return bboxes

def nms(bboxes, scores, iou_threshold):
    """ Non-Max Suppression 계산
    - bboxes: the bounding box coordinate
    - scores: scores for each bounding box
    Returns: a list containing the best indices out of a set of overlapping bbox
    """
    xmin, xmax = bboxes[:, 0], bboxes[:, 2]
    ymin, ymax = bboxes[:, 1], bboxes[:,3]
    areas = (xmax - xmin + 1) * (ymax - ymin + 1)

    score_indices = np.argsort(scores, kind="mergesort", axis=-1)[::-1] ## 점수가 높은 bounding box부터 사용
    zero = 0.0
    candidates = []

    while score_indices.size > 0: ## 처음부터 score indices를 줄여 나가면서 x축 왼쪽 오른쪽
        # 그리고 y축 위 아래의 bounding box끝 위치를 계산한다.
        i = score_indices[0]
        candidates.append(i)
        xxmax = np.maximum(xmin[i], xmin[score_indices[1:]])
        yymax = np.maximum(ymin[i], ymin[score_indices[1:]])
        xxmin = np.minimum(xmax[i], xmax[score_indices[1:]])
        yymin = np.minimum(ymax[i], ymax[score_indices[1:]])

        W = np.maximum(zero, xxmin - xxmax)
        H = np.maximum(zero, yymin - yymax)

        overlap = W * H ## (교집합)
        remain = areas[score_indices[1:]] ## remaining areas (차집합)
        aou = areas[i] + remain - overlap ## area of union
        IoU = overlap / aou

        indices = np.where(IoU <= iou_threshold)[0]
        score_indices = score_indices[indices+1]
    
    return candidates ## bounding box의 좌표 자체가 아니라 후보 index를 보내주면
    # 나중에 후보군 anchor box들을 사용해서 text proposal connector을 사용해서 연결되는 
    # text line을 구할 수 있도록 한다.

def decode(predicted_bboxes, anchor_boxes):
    anchor_height = anchor_boxes[:, 3] - anchor_boxes[:, 1] + 1. ## 원래 anchor box의 높이 
    anchor_center_y = (anchor_boxes[:, 1] + anchor_boxes[:, 3]) / 2. ## 원래 anchor box의 중심 y좌표
    truth_center_y = predicted_bboxes[..., 0] * anchor_height + anchor_center_y ## 예측된 bbox의 y축의 중심 좌표
    truth_height = np.exp(predicted_bboxes[..., 1]) * anchor_height ## 예측된 bbox의 높이

    x1 = anchor_boxes[:, 0] ## Min X (정해져 있는 anchor box의 왼쪽 x좌표를 의미)
    y1 = truth_center_y - truth_height / 2. ## Min Y
    x2 = anchor_boxes[:, 2] ## Max X (정해져 있는 anchor box의 오른쪽 x좌표를 의미)
    y2 = truth_center_y + truth_height / 2. ## Max Y
    
    bboxes = np.stack([x1, y1.squeeze(), x2, y2.squeeze()], axis = 1)

    return bboxes


class TextDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        if isinstance(cfg, dict):
            self.CONF_SCORE=cfg['CONF_SCORE']
            self.IOU_THRESH=cfg['IOU_THRESH']
            self.FEATURE_STRIDE=cfg['FEATURE_STRIDE']
            self.ANCHOR_SHIFT=cfg['ANCHOR_SHIFT']
            self.ANCHOR_HEIGHTS=cfg['ANCHOR_HEIGHTS']
        else:
            self.CONF_SCORE = cfg.CONF_SCORE ## 0.9
            self.IOU_THRESH = cfg.IOU_THRESH ## 0.2
            self.FEATURE_STRIDE = cfg.FEATURE_STRIDE ## 16
            self.ANCHOR_SHIFT = cfg.ANCHOR_SHIFT ## 16
            self.ANCHOR_HEIGHTS=cfg.ANCHOR_HEIGHTS
        self.text_proposal_connector = TextProposalConnector(self.cfg)

    def __call__(self, predictions, image_size):
        H, W = image_size
        predicted_bboxes, predicted_scores = predictions ## regr, cls
        predicted_scores = torch.softmax(predicted_scores,dim=2)
        predicted_bboxes = predicted_bboxes.cpu().numpy()
        predicted_scores = predicted_scores.cpu().numpy()

        feature_map_size = [int(np.ceil(H / self.ANCHOR_SHIFT)), int(np.ceil(W / self.ANCHOR_SHIFT))]
        anchor_boxes = generate_all_anchor_boxes(
            feature_map_size = feature_map_size,
            feature_stride = self.FEATURE_STRIDE,
            anchor_heights = self.ANCHOR_HEIGHTS,
            anchor_shift = self.ANCHOR_SHIFT
        )

        decoded_bboxes = decode(predicted_bboxes=predicted_bboxes, anchor_boxes=anchor_boxes)
        clipped_bboxes = clip_boxes(bboxes=decoded_bboxes, image_size=image_size)

        text_class = 1
        conf_scores = predicted_scores[0, :, text_class]
        conf_scores_mask = np.where(conf_scores > self.CONF_SCORE)[0] ## np.where을 사용해서 일정 점수 이상인 index의 값을 구해준다.
        
        selected_bboxes = clipped_bboxes[conf_scores_mask, :]
        selected_scores = predicted_scores[0, conf_scores_mask, text_class]
        
        candidates = nms(bboxes=selected_bboxes,
                    scores=selected_scores,
                    iou_threshold=self.IOU_THRESH
        )
        selected_bboxes, selected_scores = selected_bboxes[candidates], selected_scores[candidates]
        text_lines, scores = self.text_proposal_connector.get_text_lines(
            text_proposals=selected_bboxes,
            scores=selected_scores,
            image_size=image_size
        )

        return text_lines, scores




def to_gray(image: np.ndarray):
  return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def sobel_gradient(gray_image: np.ndarray):
  blurred = cv2.GaussianBlur(gray_image, (9, 9), 0)
  # sobel gradient
  gradX = cv2.Sobel(blurred, ddepth = cv2.CV_32F, dx=1, dy=0)
  gradY = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=0, dy=1)

  gradient = cv2.subtract(gradX, gradY)
  gradient = cv2.convertScaleAbs(gradient)

  # thresh and blur
  blurred = cv2.GaussianBlur(gradient, (9, 9), 0)
  return cv2.threshold(blurred, thresh=100, maxval=255, type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1], blurred

def morphology(threshed_image: np.ndarray, erode_iterations: int, dilate_iterations: int):
  H, W = threshed_image.shape
  if H > W:
    kernel_size = (int(W / 18), int(H/40))
  else:
    kernel_size = (int(W / 40), int(H / 18))
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
  
  morpho_image = cv2.morphologyEx(threshed_image, cv2.MORPH_CLOSE, kernel)
  morpho_image = cv2.erode(morpho_image, None, iterations=erode_iterations)
  morpho_image = cv2.dilate(morpho_image, None, iterations=dilate_iterations)

  return morpho_image

def crop(morpho_image: np.ndarray, source_image: np.ndarray):
  contours, _ = cv2.findContours(morpho_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
  crops = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
  croped = []
  croped_points = []
  H, W, C = source_image.shape
  for c in crops:
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))

    H, W, C = source_image.shape
    total = H * W
    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]

    x1 = max(min(Xs), 0)
    x2 = min(max(Xs), W)
    y1 = max(min(Ys), 0)
    y2 = min(max(Ys), H)

    new_height, new_width = y2-y1, x2-x1
    if new_height < H / 4 or new_width < W / 4: ## 잘린 이미지의 가로와 세로의 길이가 일정 비율보다 작다면 그냥 crop 하지 않고 사용한다.
      break
    else:
      croped.append(source_image[y1:y1+new_height, x1:x1 + new_width])
      croped_points.append((x1, x2, y1, y2))
  if len(croped) == 0:
    croped.append(source_image)
    croped_points.append((0, W, 0, H))
  return croped, croped_points

def load_weight(weight_name, model):
  # pretrained = torch.load(os.path.join(BASE, weight_name))
  pretrained = torch.load(weight_name)
  model_weight = model.state_dict()
  if 'model_state_dict' in pretrained:
    pretrained = pretrained['model_state_dict']
  available = {key:value for (key, value) in pretrained.items() if key in model_weight and \
                    value.shape == model_weight[key].shape}
  model_weight.update(available)
  model.load_state_dict(model_weight)

  return model

def rm_empty_box(org_image, detected_boxes):
  qualified = []
  for i, bbox in enumerate(detected_boxes):
    xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    croped_image = org_image[ymin:ymax, xmin:xmax,:]
    if croped_image.size>0:
      total_pixels = np.sum(croped_image)
      avg_white_pixels = total_pixels/croped_image.size
      if avg_white_pixels < 250:
        qualified.append(i)
  qualified = np.array(qualified, dtype = np.int32)
  return qualified

def draw_box(image, bboxes, color = (0, 255, 0), thickness = 2):
  for bbox in bboxes:
    xmin, ymin, xmax, ymax = bbox
    image = cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, thickness)
  return image

def make_divisable(value):
  if value % 16 == 0:
    return value
  else:
    a = value // 16
    return (a + 1) * 16

def rescale_for_detect(image):
  C, H, W = image.shape
  
  if H < W:
    new_w = 1024
    new_h = make_divisable(int(H * (1024 / W)))
  else:
    new_h = 1024
    new_w = make_divisable(int(W * (1024 / H)))
  rescale_factor = (W / new_w, H / new_h)
  logger.info(f"NEW_W: {new_w}, NEW_H: {new_h}")
  return (new_w, new_h), rescale_factor

def preprocess(image):
  gray_image = to_gray(image)
  sobel, blurred = sobel_gradient(gray_image)
  morpho = morphology(sobel,1, 1)
  croped, points = crop(morpho, image)
  return croped, points

class DetectBot(object):
  def __init__(self, cfg, remove_white=False):
    self.crop = remove_white
    self.cfg = cfg
    model_path = self.cfg['CKPT']
    model = CTPN().cuda()
    model = load_weight(model_path, model)
    self.model = model
    self.detector = TextDetector(self.cfg)

  def predict(self, diff_h, diff_w, image: np.ndarray):
    H, W, C = image.shape
    original_image = image
    if H > W:
      new_shape = (2048, 1024) ## (H, W)
    else:
      new_shape = (1024, 2048)
    rescale_factor = (W / new_shape[1], H / new_shape[0])
    image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(new_shape),
        transforms.Normalize(mean = MEAN, std = STD)
      ])(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
      image = image.cuda()
      reg, cls = self.model(image)

    detected_boxes, scores = self.detector((reg, cls), image_size = new_shape)
    print(len(detected_boxes)) ## Morphology를 수행해서 text영역을 최대 3부분을 찾을 수 있도록 한다.
    ratio_w, ratio_h = rescale_factor
    size_ = np.array([[ratio_w, ratio_h, ratio_w, ratio_h]])
    detected_boxes *= size_
    detected_boxes += np.array([[diff_w, diff_h, diff_w, diff_h]])
    # detected_boxes = detected_boxes[rm_empty_box(original_image, detected_boxes)]

    return detected_boxes

  def _filter_groups(self, center_boxes):
    groups = defaultdict(list)
    cnt = 0
    group_cnt = 0
    while (cnt < len(center_boxes)):
      temp_group = []
      prev = cnt
      temp_box = center_boxes[cnt]
      more = cnt
      while True:
        if more == len(center_boxes):
          break
        more_box = center_boxes[more]
        if more_box[1] - temp_box[1] <= 5:
          more += 1
          temp_box = more_box
        else:
          break
      for i in range(cnt, more):
        temp_group.append(i)
      groups[group_cnt] = temp_group
      group_cnt += 1
      if more == prev:
        cnt = more +1
      else:
        cnt = more

    return groups

  def _select_box(self, all_box, image):
    H, W, C = image.shape
    center_box = []
    for box in all_box:
      cx = (box[2] + box[0]) / 2
      cy = (box[3] + box[1]) / 2
      w, h = box[2] - box[0], box[3] - box[1]
      center_box.append([cx, cy, w, h])
    center_box = sorted(center_box, key = lambda x: x[1])

    groups = self._filter_groups(center_box) ## 가로축 기준으로 비슷한 위치에 있으면 같은 그룹으로 묶었음
    
    def return_box(center_point):
      cx, cy, w, h = center_point
      min_x, min_y = cx - w// 2, cy - h//2
      max_x, max_y = cx + w//2, cy + h//2
      return [min_x, min_y, max_x, max_y]

    ## TODO: 여기부터 영수증 형태에 맞춰서 개선해 주어야 한다. ##
    word_dict = []
    for g in groups:
      group=groups[g]
      if len(group) > 3:
        group = [center_box[x] for x in group]
        group = sorted(group, key = lambda x: (x[0], x[2])) ## 가장 왼쪽부터 오른쪽까지 정렬을 해 준다.
        
        name_box = return_box(group[0])
        quantity_box = return_box(group[1])
        word_dict.append([name_box, "name"])
        word_dict.append([quantity_box, "quantity"])
        # word_dict.append({"name": name_box,  "quantity": quantity_box})
    return word_dict




  def __call__(self, image):
    if isinstance(image, str) == False:
      image = image
    else:
      image = cv2.imread(image)
    original_image = image.copy()
    org_h, org_w, org_c = original_image.shape
    if self.crop:
      croped, points = preprocess(image)
      print(len(points))
     # if len(points) > 1:
      #  croped = [image];points=[(0, org_w, 0, org_h)]
    
    else:
      croped = [image];points = [(0, org_w, 0, org_h)];
    all_box = []
    for idx, croped_image in enumerate(croped):
      croped_point = points[idx]
      detected_boxes = self.predict(diff_h=croped_point[2], diff_w=croped_point[0], image=croped_image)
      all_box.extend(detected_boxes)
      original_image = draw_box(original_image, detected_boxes)

    selected_box = self._select_box(all_box, image)
    return original_image, all_box, image, selected_box




if __name__ == "__main__":
  import os
  class CFG:
    def __init__(self):
      self.MIN_V_OVERLAP = 0.7
      self.MIN_SIZE_SIM = 0.7
      self.MAX_HORI_GAP=20
      self.CONF_SCORE=0.9
      self.IOU_THRESH=0.2
      self.ANCHOR_SHIFT= 16
      self.FEATURE_STRIDE=16
      self.ANCHOR_HEIGHTS=[
             11, 15, 22, 32, 45, 65, 93, 133, 190, 273
        ]
      self.MIN_SCORE=0.9
      self.NMS_THRESH=0.3
      self.REFINEMENT= False
      self.LOSS_LAMDA_REG=2.0
      self.LOSS_LAMDA_CLS=1.0
      self.LOSS_LAMBDA_REFINE=2.0
      self.ANCHOR_IGNORE_LABEL=-1
      self.ANCHOR_POSITIVE_LABEL= 1
      self.ANCHOR_NEGATIVE_LABEL=0
      self.IOU_OVERLAP_THRESH_POS=0.5
      self.IOU_OVERLAP_THRESH_NEG= 0.3
  cfg = CFG()
  BASE = os.path.dirname(os.path.abspath(__file__))
  # MODEL_PATH=os.path.join(BASE, 'demo', 'weight', 'CTPN_FINAL_CHECKPOINT.pth')
  FILE_PATH='/home/guest/speaking_fridgey/ocr_exp_v2/text_detection/demo/sample/recipt13.png'
  #detected = detect(cfg, FILE_PATH, MODEL_PATH)
  import yaml
  with open('/home/guest/speaking_fridgey/ocr_deploy_v1/inference_server/src/detector/detector.yaml', 'r') as f:
    detect_cfg = yaml.load(f, Loader=yaml.FullLoader)
  bot = DetectBot(detect_cfg, remove_white=True)
  detected,box, image = bot(FILE_PATH)
  cv2.imwrite(FILE_PATH.split('.')[0] + '_result' + '.jpg', detected)
 # os.chdir()
 # drawn_image = detect()