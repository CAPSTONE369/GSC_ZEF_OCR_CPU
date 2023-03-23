import os, sys
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.getcwd()))
from .text_detection.ctpn import CTPN
from .text_detection.ctpn_connector import TextProposalConnector
from .text_detection.ctpn_anchor import generate_all_anchor_boxes
from .text_detection.detection_utils import nms, decode, clip_boxes
import numpy as np
import torch
from torchvision import transforms
import easyocr
import cv2
import yaml
import math
import copy

BASE_PATH=os.path.dirname(os.path.abspath(__file__)) # os.getcwd()
PRETRAINED_PATH=os.path.join(BASE_PATH, 'pretrained_models')

def recipt_preprocessing(image):
    original_image = image.copy()

    # (1) Change to binary image
    # image[np.where(image < image.mean())] = 0
    grey_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret, binary_image = cv2.threshold(grey_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # (2) Closing Transformation
    kw = 1;kh = image.shape[0] // image.shape[1]
    kh = kh if kh % 2 == 1 else kh + 1
    kernel = np.ones((kh, kw))
    morpho_image = cv2.dilate(binary_image, kernel, iterations = 5)
    morpho_image = cv2.erode(morpho_image, np.ones((5,5)), iterations = 5)
    cpy = copy.deepcopy(morpho_image)

    # (3) Find contours to detect the area of the recipt
    contours, _ = cv2.findContours(cpy, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    contour_image = cv2.drawContours(morpho_image, contours, contourIdx=-1, color=255, thickness=3)
    contour_area = sorted([(cv2.contourArea(cnt), idx) for idx, cnt in enumerate(contours)], reverse=True)

    rect = cv2.minAreaRect(contours[contour_area[0][1]]) # 제일 큰 영역에 대해서 사용하기로 한다.
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # (4) Get the perspective transformation
    box = sorted(box, key = lambda x: (x[0], x[1])) # 좌상 -> 좌하 -> 우상 -> 우하
    trans_w = np.array(box).T[1]
    W = max(trans_w) - min(trans_w)
    trans_h = np.array(box).T[0]
    H = max(trans_h) - min(trans_h)
    pts1 = np.float32(box)
    pts2 = np.float32([[0,0],[0,W],[H,0],[H,W]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(original_image, M, (H, W))

    return dst

def remove_empty(text_lines):
    new_lines =[]
    for text_line in text_lines:
        x1,y1,x2,y2= text_line
        if (math.floor(x1) == math.ceil(x2) or math.floor(y1) == math.ceil(y2)):
            continue
        new_lines.append(text_line)
    return new_lines
            

def temporary_kie(text_lines, image):
    lang = ["ko", "en"] # easy ocr에서 사용 가능한 언어
    reader = easyocr.Reader(lang_list=lang, gpu=True, detect_network="craft", recognizer=True, detector=True)
    grey_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    free_list_agg = [[]]
    name, quant = [], []
    # (1) get text recognition results from the detection box
    for line in text_lines:
        x1, y1 = math.floor(line[0]), math.floor(line[1])
        x2, y2 = math.ceil(line[2]), math.ceil(line[3])
        box, word, score = reader.recognize(grey_image, horizontal_list=[[x1, x2, y1, y2]], free_list=free_list_agg[0])[0]
        word = re.sub(re.compile("[^가-힣]"), "", word)
    # (2) Find '수량' and '상품명' from the recognized words
        if '상품' in word or "품명" in word:
            goods.append(line)
        if "수량" in word:
            quant.append(line)
    # (3) Check all the words in the similar horizontal line with '수량' and '상품명' 
    # (4) Return them as a pair
    pass

class TextDetector(object):
    def __init__(self, detect_config, pretrained_model):
        super(TextDetector, self).__init__()
        self.cfg=detect_config
        # self.device = torch.device('cpu')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pretrained_model=pretrained_model
        self.CONF_SCORE=self.cfg['CONF_SCORE']
        self.IOU_THRESH=self.cfg['IOU_THRESH']
        self.FEATURE_STRIDE=self.cfg['FEATURE_STRIDE']
        self.ANCHOR_SHIFT=self.cfg['ANCHOR_SHIFT']
        self.ANCHOR_HEIGHTS=self.cfg['ANCHOR_HEIGHTS']
        self.PREPROCESS=False  # self.cfg['PREPROCESS']
    
        self.text_proposal_connector=TextProposalConnector(detect_config)
        self.load_model()

    def load_model(self):
        model = CTPN() # .to(self.device)
        # weight_path = os.path.join(PRETRAINED_PATH, self.cfg['PRETRAINED'])
        weight_path = self.pretrained_model
        assert os.path.isfile(weight_path)
        weight = torch.load(weight_path, map_location = self.device)

        if 'model_state_dict' in weight:
            weight = weight['model_state_dict']
        weight = {
            key:value for key, value in weight.items() if key in model.state_dict() and \
                value.shape == model.state_dict()[key].shape
        }
        model.load_state_dict(weight)

        self.model = model.to(self.device)
        self.model.eval()
        print("MODEL LOADED")

    def detect(self, image):
        detect_meta = self.get_output_from_model(image)
        text_lines, scores, original_image = self.change_output_to_text_line(detect_meta)

        return text_lines, original_image

    def get_output_from_model(self, image):
        """ Args
        image: image must be a numpy array
        Output: (predicted_bboxes, predicted_scores) -> These are the predictions of the CTPN model
        """
        # if self.PREPROCESS:
        #     image = recipt_preprocessing(image)
        H, W, C = image.shape
        input_image = image
        if H > W:new_shape = (2048, 1024)
        else:new_shape=(1024, 2048)
        rescale_factor = (W / new_shape[1], H / new_shape[0]) # [ratio_w, ratio_h]
        tensor_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(new_shape),
            transforms.Normalize(self.cfg['MEAN'], self.cfg['STD'])
        ])(image)
        tensor_image = tensor_image.unsqueeze(0)

        with torch.no_grad():
            tensor_image = tensor_image.to(self.model.get_device())
            reg, cls = self.model(tensor_image)
        
        detect_meta = {
            "regression": reg,
            "classification": cls,
            "H": new_shape[0], "W": new_shape[1],
            "ratio_w": rescale_factor[0], "ratio_h": rescale_factor[1],
            "original_image": input_image
        }
        return detect_meta

    def change_output_to_text_line(self, detect_meta):
        predicted_bboxes, predicted_scores = detect_meta['regression'], detect_meta['classification']
        predicted_scores = torch.softmax(predicted_scores, dim=2)
        predicted_bboxes = predicted_bboxes.cpu().numpy()
        predicted_scores = predicted_scores.cpu().numpy()

        feature_map_size = [
            int(np.ceil(detect_meta["H"] / self.ANCHOR_SHIFT)), int(np.ceil(detect_meta["W"] / self.ANCHOR_SHIFT))
        ]
        anchor_boxes = generate_all_anchor_boxes(
            feature_map_size = feature_map_size,
            feature_stride=self.FEATURE_STRIDE,
            anchor_heights=self.ANCHOR_HEIGHTS,
            anchor_shift=self.ANCHOR_SHIFT
        )

        decoded_bboxes = decode(predicted_bboxes=predicted_bboxes, anchor_boxes=anchor_boxes)
        clipped_bboxes = clip_boxes(bboxes=decoded_bboxes, image_size=(detect_meta["H"], detect_meta["W"]))
        
        text_class = 1
        conf_scores = predicted_scores[0, :, text_class]
        conf_scores_mask = np.where(conf_scores > self.CONF_SCORE)[0] ## np.where을 사용해서 일정 점수 이상인 index의 값을 구해준다.
        
        selected_bboxes = clipped_bboxes[conf_scores_mask, :]
        selected_scores = predicted_scores[0, conf_scores_mask, text_class]

        candidates = nms(
            bboxes=selected_bboxes,
            scores=selected_scores,
            iou_threshold=self.IOU_THRESH
        )

        selected_bboxes, selected_scores = selected_bboxes[candidates], selected_scores[candidates]
        text_lines, scores = self.text_proposal_connector.get_text_lines(
            text_proposals=selected_bboxes,
            scores=selected_scores,
            image_size=(detect_meta["H"], detect_meta["W"])
        )
        text_lines *= np.array([[
            detect_meta['ratio_w'], detect_meta['ratio_h'], detect_meta['ratio_w'], detect_meta['ratio_h']
        ]])
        # text_lines = remove_empty(text_lines)
        return text_lines, scores, detect_meta["original_image"]

        
    

if __name__ == "__main__":
    sample_image = cv2.cvtColor(cv2.imread('/home/guest/speaking_fridgey/fridgeyocr/fridgeyocr/demo/recipt3.jpg'), cv2.COLOR_BGR2RGB)
    with open('/home/guest/speaking_fridgey/fridgeyocr/fridgeyocr/config/ctpn_detection.yaml', 'r') as f:
        detect_config = yaml.load(f, Loader = yaml.FullLoader)
    detector = TextDetector(detect_config)
    text_lines = detector.detect(sample_image)
    print(len(text_lines))

