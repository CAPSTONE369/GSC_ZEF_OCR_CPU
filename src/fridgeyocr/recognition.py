from .text_recognition.jamo_utils.jamo_merge import join_jamos
from .text_recognition.hennet_label_converter import HangulLabelConverter
from .text_recognition.HENNET import HENNet

import torch
import os
import easyocr
import cv2
import math
from torchvision import transforms
from scipy.ndimage import zoom
import re
from collections import defaultdict

BASE_PATH=os.path.dirname(os.path.abspath(__file__))  # os.getcwd()
PRETRAINED_PATH=os.path.join(BASE_PATH, 'pretrained_models')


class TextRecognizer(object):
    def __init__(self, recog_config, pretrained_model):
        super(TextRecognizer, self).__init__()
        self.cfg = recog_config
        self.pretrained_model=pretrained_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        self.easy_ocr =  easyocr.Reader(lang_list=["ko", "en"], gpu=True, detect_network="craft", recognizer=True, detector=True)
        self.label_converter = HangulLabelConverter(
            add_num=self.cfg['ADD_NUM'], add_eng=self.cfg['ADD_ENG'],
            add_special=self.cfg['ADD_SPECIAL'], max_length=self.cfg['MAX_LENGTH']
        )
        self.load_model()
    
    def load_model(self):
        model = HENNet(
            img_w=self.cfg["IMG_W"], img_h=self.cfg['IMG_H'], res_in=self.cfg["RES_IN"],
            encoder_layer_num=self.cfg["ENCODER_LAYER_NUM"], attentional_transformer=self.cfg['ATTENTIONAL_TRANSFORMER'],
            class_n=self.label_converter._get_class_n(), adaptive_pe=self.cfg['ADAPTIVE_PE'], 
            seperable_ffn=self.cfg['SEPERABLE_FFN'], head_num=self.cfg['HEAD_NUM'],
            batch_size=self.cfg['BATCH_SIZE'], use_conv=self.cfg['USE_CONV'],
            embedding_dim=self.cfg['EMBEDDING_DIM']
        )
        model.to(self.device)
        # pretrained_dir = os.path.join(PRETRAINED_PATH, self.cfg['PRETRAINED'])
        # model.load_state_dict(torch.load(pretrained_dir))
        model.load_state_dict(torch.load(self.pretrained_model))
        model.eval()

        self.model = model
    
    def make_model_input(self, croped_image):
    
        new_w = self.cfg['IMG_W']
        new_h = self.cfg['IMG_H']
        org_w, org_h = croped_image.shape[1], croped_image.shape[0]
        zoomed_image = zoom(croped_image, [new_h/org_h, new_w/org_w])
        tensor_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.cfg['MEAN'], self.cfg['STD'])
        ])(zoomed_image)

        return tensor_image

    def get_easyocr_output(self, image, horizontal_list, free_list_agg=[[]]):
        predicted = self.easy_ocr.recognize(image, horizontal_list=horizontal_list, free_list=free_list_agg[0])[0]
        box, word, score = predicted[0], predicted[1], predicted[2]
        print(word)
        han_word = re.sub(re.compile('[^가-힣]'), '', word)
        word = re.sub(re.compile('[^가-힣0-9]'), '', word)
        if word == "":
            return ["dummy", word]
        else:
            if "상품" in han_word or "품명" in word:
                print(word)
                return ["name_header", word]
            if "수량" in han_word:
                print(word)
                return ["quant_header", word]
            if han_word == "" and word != "":
                return ["quant_candidate", word]
            if han_word != "":
                return ["name_candidate", word]
            return ["dummy", word]
        
    def recognize(self, image, text_lines):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        name_header = []; quant_header = [];
        new_bbox = []
        text_dict = {}
        
        answer = []

        for idx, text_line in enumerate(text_lines):
            x1, y1 = math.floor(text_line[0]), math.floor(text_line[1])
            x2, y2 = math.ceil(text_line[2]), math.ceil(text_line[3])
        
            croped_box = image[y1:y2, x1:x2]
            
            with torch.no_grad():
                input = self.make_model_input(croped_box)
                input = input.unsqueeze(0)
                input = input.to(self.model.get_device())
                pred = self.model(input)
                pred_text, _, _ = self.label_converter.decode(pred)
                easyocr_pred = self.get_easyocr_output(image, [[x1, x2, y1, y2]])
                if easyocr_pred[0] == "name_header":
                    name_header = [x1, y1, x2, y2]
                if easyocr_pred[0] == "quant_header":
                    quant_header = [x1, y1, x2, y2]
                new_bbox.append([x1, y1, x2, y2, idx])
                text_dict[idx] = {
                    "text": pred_text, "box": [x1, y1, x2, y2],
                    "easy_ocr_pred": easyocr_pred,
                }
        new_sorted_bbox = sorted(new_bbox, key = lambda x: (x[1], x[0]))
        if name_header != []:
            hx1, hy1, hx2, hy2 = name_header
            width = hx2-hx1
            for _, td in text_dict.items():
                box = td["box"]
                if box[1] < hy2: # <상품명> 보다 아래에 위치한 text box이어야만 한다.
                    continue
                if td['easy_ocr_pred'][0] != 'name_candidate': # 한글이 무조건 포함이 되어야 상품명으로 구분할 수 있다.
                    continue 
                mx = max(hx1, box[0])
                Mx = min(hx2, box[2])
                inter_w = Mx-mx
                union_w = max(hx2, box[2]) - min(hx1, box[0])


                if (inter_w <= (union_w * 0.8) and width <= (box[2] - box[0])): # horizontal 뱡향으로 0.8만큼 이상 겹치는 경우를 사용한다.
                    answer.append(
                        {"name": td['easy_ocr_pred'][1], # td["text"], 
                         "quantity": str(1)}
                    )
        else:
            """text box의 가로축에 있는 것들의 개수를 기반으로 + 위치 기반으로 3~4개의 text box가 한줄에 있으면 <관심 정보>일것이라고 가정"""
            answer = self._select_box(new_bbox, image, text_dict)

        return answer
    
    def _select_box(self, all_box, image, text_dict):
        H, W, C = image.shape
        center_box = []
        for box in all_box:
            cx = (box[2] + box[0]) / 2
            cy = (box[3] + box[1]) / 2
            w, h = box[2] - box[0], box[3] - box[1]
            center_box.append([cx, cy, w, h, box[-1]])
        center_box = sorted(center_box, key = lambda x: x[1])

        groups = self._filter_groups(center_box) ## 가로축 기준으로 비슷한 위치에 있으면 같은 그룹으로 묶었음
    
        def return_box(center_point):
            cx, cy, w, h = center_point
            min_x, min_y = cx - w// 2, cy - h//2
            max_x, max_y = cx + w//2, cy + h//2
            return [min_x, min_y, max_x, max_y]

        answer = []
        for g in groups:
            group=groups[g]
            if len(group) > 3:
                group = [center_box[x] for x in group]
                group = sorted(group, key = lambda x: (-x[2], x[0])) ## 가장 왼쪽부터 오른쪽까지 정렬을 해 준다.
        
                name_box = return_box(group[0])
                group = sorted(group[1:], key = lambda x: (x[2], -x[0]))
                quantity_box = return_box(group[0])
                answer.append({
                    "name": text_dict[name_box[-1]]["easy_ocr_pred"][1], # text_dict[name_box[-1]]["text"],
                    "quantity": text_dict[quantity_box[-1]]["text"]
                })

        return answer
    
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

                
        