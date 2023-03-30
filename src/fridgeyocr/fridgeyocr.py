from .detection import TextDetector # get_detector
from .recognition import TextRecognizer # get_recongizer
from .data_parse_distance import get_roi_of_recipt
from .data_parse import get_roi_with_keywords, get_name_in_roi
import cv2
import numpy as np
import torch
import os
import sys
from pathlib import Path
from loguru import logger
import yaml

# os.environ['CUDA_VISIBLE_DEVICES']="6" # 6번 GPU만 사용 가능 -> 이건 guest 서버일때만
"""
configuration file을 만들기 위한 규칙
fridegeyocr/config 폴더 안에 무조건 들어 있어야 하며, {MODEL_NAME}_{TASK_NAME}.yaml의 파일명 규칙을 지켜야 한다.
"""
CURRENT_PATH=os.path.dirname(os.path.abspath(__file__))
CONFIG_FOLDER=os.path.join(CURRENT_PATH, "config")

def read_yaml(yaml_path: str):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.load(f, Loader = yaml.FullLoader)
    return data

def load_detector(detect_network, detect_network_pretrained):
    config_path=os.path.join(CONFIG_FOLDER, f"{detect_network}_detection.yaml")
    assert os.path.isfile(config_path) == True
    detect_config = read_yaml(config_path)
    detector = TextDetector(detect_config, detect_network_pretrained)

    return detector



def load_recognizer(recog_network, recog_network_pretrained):
    config_path=os.path.join(CONFIG_FOLDER, f"{recog_network.lower()}_recognition.yaml")
    assert os.path.isfile(config_path) == True
    recog_config = read_yaml(config_path)
    recognizer = TextRecognizer(recog_config, recog_network_pretrained)
    
    return recognizer

class Reader(object):
    def __init__(self, 
                 detect_network_pretrained, 
                 recog_network_pretrained, 
                 gpu=True, detect_network='ctpn', 
                 recog_network='hennet', detector=True, 
                 recognizer=True, verbose=True):
        super(Reader, self).__init__()

        if gpu is False:
            self.device = 'cpu'
            if verbose:
                logger.info("Using CPU. This operation wil run faster with a GPU")
        elif not torch.cuda.is_available():
            self.device = 'cpu'
            if verbose:
                logger.info("Unable to use GPU. Running on CPU")
        else:
            self.device = 'cuda'
        self.recognizer = load_recognizer(recog_network, recog_network_pretrained) if recognizer else None
        self.detector = load_detector(detect_network,detect_network_pretrained) if detector else None
    
    def __call__(self, image):
        if self.detector is not None:
            print("DETECTION START")
            text_lines, image = self.detector.detect(image)
            print(f"DETECTION END: {len(text_lines)}")

        if self.recognizer is not None:
            print("RECOGNITION START")
            answer = self.recognizer.recognize(image, text_lines)
            print(f"RECOGNITION_END: {len(answer)}")
            # print(answer)
            # left, right, drawn = get_roi_with_keywords(answer, image)
            left, right, drawn = get_roi_of_recipt(answer, image)
            answer = get_name_in_roi(left, right, answer)
            return answer
        else:
            return text_lines
    
    def _detect(self, image):
        assert isinstance(image, np.ndarray) == True
        if self.detector is None:
            print("INVALID OPERATION - DEFINE THE DETECTOR FIRST")
            return
        text_lines = self.detector.detect(image)
        return text_lines
    


        
        
        


