""" predict.py
- 입력 이미지를 받아서 (이때 text box단위로 잘린 이미지여야 한다.)
- model의 sqeuence length X class number의 길이의 prediction vector을 바탕으로 실제 글자로 변환
- 한글, 영어, 숫자는 예측 그대로 사용하고 '[UNK]'이면 공백으로 남겨 놓음
"""

import os, sys, torch, cv2, torch, math
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model.hen_net import HENNet
from torchvision import transforms
from label_converter import HangulLabelConverter, GeneralLabelConverter

class HENNETPredictBot(object):
    def __init__(self, recog_cfg):
        super().__init__()
        self.recog_cfg = recog_cfg
        self.device = torch.device('cuda:6' if torch.cuda.is_available else 'cpu')
        if self.recog_cfg['CONVERTER'].upper() == 'GENERAL':
            max_seq_length = self.recog_cfg['MAX_LENGTH'] 
            self.converter = GeneralLabelConverter(max_length = self.recog_cfg['MAX_LENGTH'])
        elif self.recog_cfg['CONVERTER'].upper() == 'HANGUL':
            max_seq_length = self.recog_cfg['MAX_LENGTH'] * 3
            self.converter = HangulLabelConverter(
                add_num=self.recog_cfg['ADD_NUM'], add_eng=self.recog_cfg['ADD_ENG'],
                max_length=self.recog_cfg['MAX_LENGTH'] * 3
            )
        else:
            raise NotImplementedError
        self.model = HENNet(
            img_w=self.recog_cfg['IMG_W'], img_h=self.recog_cfg['IMG_H'], 
            max_seq_length=max_seq_length,
            adaptive_pe=self.recog_cfg['ADAPTIVE_PE'],
            res_in=self.recog_cfg['RES_IN'], head_num=recog_cfg['HEAD_NUM'], 
            encoder_layer_num=recog_cfg['ENCODER_LAYER_NUM'],
            tps=False, activation=self.recog_cfg['ACTIVATION'], 
            rgb=self.recog_cfg['RGB'],
            use_resnet=self.recog_cfg['USE_RESNET'],
            batch_size=1, seperable_ffn=self.recog_cfg['SEPERABLE_FFN'], 
            class_n=len(self.converter.char_decoder_dict)
        ).to(self.device)
        self.model.load_state_dict(
            torch.load(self.recog_cfg['CKPT'])
        )


    def predict_single(self,image):
        h, w, c = image.shape
        if c == 3 and self.recog_cfg['RGB'] == False:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.recog_cfg['MEAN'], std=self.recog_cfg['STD']),
            transforms.Resize((self.recog_cfg['IMG_H'], self.recog_cfg['IMG_W']))
        ])(image)
        if (len(image.shape) == 3):
            image = torch.unsqueeze(image, dim=0)
        image = image.to(self.device)
        #print(image.shape)
        self.model.eval()

        pred = self.model(image, batch_size=1)
        pred_text, pred_score, pred_length = self.converter.decode(pred)

        return pred_text
        
    def predict_one_call(self, image_dict:dict):
        predictions = {}
        for key, value in image_dict.items():
            pred_str = self.predict_single(value)
            # predictions[key] = {'text': pred_str, 'bbox':[]}
            predictions[key] = {'text': pred_str[0]}
            print(pred_str[0])
        return predictions