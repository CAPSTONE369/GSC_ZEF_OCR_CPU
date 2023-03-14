from src.fridgeyocr import fridgeyocr
import os, yaml, math, cv2
from PIL import Image
import numpy as np
from flask import request, jsonify
BASE_PATH=os.path.dirname(os.path.abspath(__file__))
from flask import Flask
app = Flask(__name__)

PRETRAINED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src/fridgeyocr/pretrained_models')
def run_ocr(
    detection_cfg: dict, 
    input_image,
    # image_path, ## (우선은 경로 사용) 이미지는 array의 형태로 flask 서버에서 받아올 것이다.
    # detection_model_path: str, ## 사전학습된 CTPN모델 경로,
    remove_white: bool, ## text detection을 수행하기 위해서 주변 테두리를 자르는 전처리 과정을 거칠 것인가
    # recognition_model_path: str, ## 사전학습된 HangulNet 모델 경로
    recognition_cfg: dict,
    ):
    detection_model_path = os.path.join(PRETRAINED_DIR, detection_cfg['PRETRAINED'])
    recognition_model_path = os.path.join(PRETRAINED_DIR, recognition_cfg['PRETRAINED'])
    reader = fridgeyocr.Reader(
        detect_network_pretrained = detection_model_path,
        recog_network_pretrained = recognition_model_path,
        gpu = True, detect_network = 'ctpn', recog_network = 'hennet'
    )
    answer = reader(image=input_image)
    return answer
    

@app.route("/")
def main():
    return "Hello World" 


@app.route("/model", methods=['POST'])
def model():
    if request.method == 'POST':
        file = request.files['file']
        image = Image.open(file.stream)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        # recipt_image = request.files['image'] ## flutter 서버에서 이미지 받아오기 (np array 형태라고 가정하자)
        # image_path = '/home/guest/speaking_fridgey/ocr_exp_v2/text_detection/demo/sample/recipt.jpg'
        # 이미지를 로컬에 저장하는 과정이 필요함 -> 근데 서버에 베포한다고 생각하면 어떻게 이미지를 로컬에 저장하는 걸까?
        with open(os.path.join(BASE_PATH, 'src/fridgeyocr/config/hennet_recognition.yaml'), 'r') as f:
            recog_cfg = yaml.load(f, Loader=yaml.FullLoader)
        with open(os.path.join(BASE_PATH, 'src/fridgeyocr/config/ctpn_detection.yaml'), 'r') as f:
            detect_cfg = yaml.load(f, Loader=yaml.FullLoader)
        pred_dict = run_ocr(
            detection_cfg=detect_cfg, input_image=image, #  image_path=image_path,
            remove_white=True, recognition_cfg=recog_cfg
        )

    print(pred_dict)
    return pred_dict, 200 

if __name__ == "__main__":
    app.debug=True
    app.run(port='8080', host='0.0.0.0')
