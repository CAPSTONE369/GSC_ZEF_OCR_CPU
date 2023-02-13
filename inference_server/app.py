from src.detector.detector import DetectBot
from src.recognizer.recognizer import HENNETPredictBot
import os, yaml, math, cv2
from PIL import Image
import numpy as np
from flask import request, jsonify
BASE_PATH=os.path.dirname(os.path.abspath(__file__))
from flask import Flask
app = Flask(__name__)

def crop_images(image, bbox):
    croped = {}
    for idx, box in enumerate(bbox):
        x1,y1,x2,y2=box
        x1, y1 = math.ceil(x1), math.ceil(y1)
        x2, y2 = math.floor(x2), math.floor(y2)
        cv2.imwrite(f'test_{idx}.png', image[y1:y2, x1:x2,:])
        croped[idx] = image[y1:y2, x1:x2, :]
    return croped

def run_ocr(
    detection_cfg: dict, 
    input_image,
    # image_path, ## (우선은 경로 사용) 이미지는 array의 형태로 flask 서버에서 받아올 것이다.
    # detection_model_path: str, ## 사전학습된 CTPN모델 경로,
    remove_white: bool, ## text detection을 수행하기 위해서 주변 테두리를 자르는 전처리 과정을 거칠 것인가
    # recognition_model_path: str, ## 사전학습된 HangulNet 모델 경로
    recognition_cfg: dict,
    ):
    detection_model_path = detection_cfg['CKPT']
    recognition_model_path = recognition_cfg['CKPT']
    detect_bot = DetectBot(
        cfg=detection_cfg, remove_white=remove_white
    )
    detected, box, image = detect_bot(input_image) ## bounding box 그려진 원본 이미지와 detection box 좌표
    croped_dict = crop_images(image, bbox=box)
    
    if recognition_model_path is None:
        return detected ## recognition path가 없는 경우에는 그냥 감지된 영역에 bbox처리된 이미지만 return
    else:
        if recognition_cfg['NAME'] == 'CLOVA':
            recog_bot = CLOVAPredictBot(recognition_model_path)
            pred_dict = recog_bot.predict_one_call(croped_dict)
        elif recognition_cfg['NAME'] == 'HENNET':
            recog_bot = HENNETPredictBot(recognition_cfg)
            pred_dict = recog_bot.predict_one_call(croped_dict)
        #for idx, b in enumerate(box):
        #    pred_dict[idx]['bbox'] = b
        return pred_dict

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
        with open(os.path.join(BASE_PATH, 'src/recognizer/recognizer.yaml'), 'r') as f:
            recog_cfg = yaml.load(f, Loader=yaml.FullLoader)
        with open(os.path.join(BASE_PATH, 'src/detector/detector.yaml'), 'r') as f:
            detect_cfg = yaml.load(f, Loader=yaml.FullLoader)
        pred_dict = run_ocr(
            detection_cfg=detect_cfg, input_image=image, #  image_path=image_path,
            remove_white=True, recognition_cfg=recog_cfg
        )
    
    return pred_dict, 200 # jsonify(pred_dict), 200

if __name__ == "__main__":
    app.debug=True
    app.run(port='8080') # host='0.0.0.0')
