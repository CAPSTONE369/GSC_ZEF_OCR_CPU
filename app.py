from src.fridgeyocr import fridgeyocr
from src.fridgeyocr import gdrive_utils
import os, yaml, math, cv2, json
from PIL import Image
import numpy as np
from flask import request, render_template
BASE_PATH=os.path.dirname(os.path.abspath(__file__))
from flask import Flask
app = Flask(__name__)



PRETRAINED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src/fridgeyocr/pretrained_weights')
os.makedirs(PRETRAINED_DIR, exist_ok=True)
## Google Drive에서 파일 ID는 파일 위치와 파일 이름만 동일하다면 계속 일정하다.
MODEL_WEIGHTS = {
    "TPS_RESNET_BILSTM_CTC": '1XpujSr2yV35E-25Gs8FmxOJDgYthehg3',
    "BEST_TPS_RESNET_BILSTM_CTC": '130_1LqWwJgcGOP0Oo_bj1WjBuF2H9MSl', # '1Z3dP3cp2f1P5tfcreWkzf2hAfN_J0EgI', # '1BB80X_OQCOJiLYDms77fbw28KBnVwnSY', # '1Z3dP3cp2f1P5tfcreWkzf2hAfN_J0EgI',
    "CTPN": '1XF76z5iRhYxrbAiLIddJC5X5dJkmumDh'
}

def make_recognition_key(recog_cfg):
    trans = recog_cfg['TRANSFORMATION'].upper()
    feat = recog_cfg['FEATUREEXTRACTION'].upper()
    seq = recog_cfg['SEQUENCEMODELING'].upper()
    pred = recog_cfg['PREDICTION'].upper()

    return f"BEST_{trans}_{feat}_{seq}_{pred}"

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
    detection_id = MODEL_WEIGHTS['CTPN']
    recognition_id = MODEL_WEIGHTS[make_recognition_key(recognition_cfg)]

    if os.path.isfile(detection_model_path) == False:
        gdrive_utils.download_model_weight_from_gdrive(detection_id, detection_model_path)
    if os.path.isfile(recognition_model_path) == False:
        gdrive_utils.download_model_weight_from_gdrive(recognition_id, recognition_model_path)

    reader = fridgeyocr.Reader(
        detect_network_pretrained = detection_model_path,
        recog_network_pretrained = recognition_model_path,
        gpu = True, detect_network = 'ctpn', recog_network = 'crnn'
    )
    answer = reader(image=input_image)
    return answer
    

@app.route("/")
def main():
    return "Hello World" 

@app.route("/demo", methods=["POST", "GET"])
def demo():
    if request.method == 'GET':
        return render_template("demo.html")
    if request.method == 'POST':
        file = request.files['file']
        image = Image.open(file.stream)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        with open(os.path.join(BASE_PATH, 'src/fridgeyocr/config/crnn_recognition.yaml'), 'r') as f:
            recog_cfg = yaml.load(f, Loader=yaml.FullLoader)
        with open(os.path.join(BASE_PATH, 'src/fridgeyocr/config/ctpn_detection.yaml'), 'r') as f:
            detect_cfg = yaml.load(f, Loader=yaml.FullLoader)
        pred_dict = run_ocr(
            detection_cfg=detect_cfg, input_image=image, #  image_path=image_path,
            remove_white=True, recognition_cfg=recog_cfg
        )
        return render_template("result.html", result= pred_dict)




@app.route("/model", methods=['POST'])
def model():
    if request.method == 'POST':
        file = request.files['file']
        image = Image.open(file.stream)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        # recipt_image = request.files['image'] ## flutter 서버에서 이미지 받아오기 (np array 형태라고 가정하자)
        # image_path = '/home/guest/speaking_fridgey/ocr_exp_v2/text_detection/demo/sample/recipt.jpg'
        # 이미지를 로컬에 저장하는 과정이 필요함 -> 근데 서버에 베포한다고 생각하면 어떻게 이미지를 로컬에 저장하는 걸까?
        with open(os.path.join(BASE_PATH, 'src/fridgeyocr/config/crnn_recognition.yaml'), 'r') as f:
            recog_cfg = yaml.load(f, Loader=yaml.FullLoader)
        with open(os.path.join(BASE_PATH, 'src/fridgeyocr/config/ctpn_detection.yaml'), 'r') as f:
            detect_cfg = yaml.load(f, Loader=yaml.FullLoader)
        pred_dict = run_ocr(
            detection_cfg=detect_cfg, input_image=image, #  image_path=image_path,
            remove_white=True, recognition_cfg=recog_cfg
        )

    print(pred_dict)
    # res = json.loads(pred_dict)
    return json.dumps(pred_dict, ensure_ascii=False, indent=4)
    # return json.dumps(pred_dict, ensure_ascii=False, indent=4)
    # return pred_dict, 200 

if __name__ == "__main__":
    app.debug=True
    app.run(port='5000', host='0.0.0.0')

