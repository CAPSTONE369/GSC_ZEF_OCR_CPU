# 🔆 OCR Repository of Zero Fridge
**Zero Fridge** is an application which manages user's fridges by alarming of food subscription dates, and makes recording food bought easy by using the recipt OCR technology.  

**✔️ This is the repository for the OCR flow; containing of `Text Detection`, `Text Recognition`, and `Key Information Extraction`.**  

<br />

## 1. To Run Inference Of Fridge OCR Model
### 1. Environment
- If you want to test this model, you should install Docker.
- We recommend that please run this model on GPU

<br />

_**HOW TO INSTALL DOCKER?**_
- Ubuntu
```bash
sudo apt update
sudo apt install docker.io
```

- Mac, Window
> Use Docker Desktop.
  [YOU CAN INSTALL HERE](https://www.docker.com/products/docker-desktop/)

<br />

### 2. Installation
1) Pull Docker Image
```bash
sudo docker pull sunnyineverywhere/fridge-ocr-flask
```

2) Usage
You can test the OCR demonstration on your recipt image by running the `Flask` api.
```bash
sudo docker run -d -p 5000:5000 sunnyineverywhere/fridge-ocr-flask
```
- The demo is available in `localhost:5000/model` if you wish to test the result via `Postman`.
- The demo is available in `localhost:5000/demo` if you wish to test the result by uploading the image via web.

<br />

## 2. Explanation on How it Works
### 1) Procedure
<img src="./figures/fridgeyocr_flow.png">

### 2) On WEB
<table border="1" cellspacing="0" cellpadding="0" width="100%">
  <tr>
    <td width="50%" align="center">`demo` before submit</td>
    <td width="50%" align="center">`demo` after submit</td>
  </tr>
  <tr width="100%">
        <td width="50%" align="center"><img alt="image" src="https://user-images.githubusercontent.com/80109963/228731166-32423ce0-91eb-4f14-9e69-4d7673f5a630.png"></td>
        <td width="50%" align="center"><img alt="image" src="https://user-images.githubusercontent.com/80109963/228731309-8c7e1bbf-663c-4575-af57-8b29f08ad9d8.png"></td>
  </tr>
</table>


## 3. Directory Structure
```
MAIN_REPO
.
├── Dockerfile
├── README.md
├── __pycache__
│   └── app.cpython-310.pyc
├── app.py
├── requirements.txt
├── src
│   ├── detector
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-310.pyc
│   │   │   ├── anchor.cpython-310.pyc
│   │   │   ├── box_match.cpython-310.pyc
│   │   │   ├── connector.cpython-310.pyc
│   │   │   ├── ctpn.cpython-310.pyc
│   │   │   └── detector.cpython-310.pyc
│   │   ├── anchor.py
│   │   ├── box_match.py
│   │   ├── connector.py
│   │   ├── ctpn.py
│   │   ├── detector.py
│   │   └── detector.yaml
│   ├── fridgeyocr
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-310.pyc
│   │   │   ├── data_parse.cpython-310.pyc
│   │   │   ├── data_parse_distance.cpython-310.pyc
│   │   │   ├── detection.cpython-310.pyc
│   │   │   ├── fridgeyocr.cpython-310.pyc
│   │   │   ├── gdrive_utils.cpython-310.pyc
│   │   │   └── recognition.cpython-310.pyc
│   │   ├── config
│   │   │   ├── __init__.py
│   │   │   ├── crnn_recognition.yaml
│   │   │   ├── ctpn_detection.yaml
│   │   │   └── hennet_recognition.yaml
│   │   ├── data_parse.py
│   │   ├── data_parse_distance.py
│   │   ├── detection.py
│   │   ├── fridgeyocr.py
│   │   ├── gdrive_utils.py
│   │   ├── pretrained_weights
│   │   │   ├── crnn_pretrained.pt
│   │   │   └── ctpn_pretrained.pt
│   │   ├── recognition.py
│   │   ├── text_detection
│   │   │   ├── __init__.py
│   │   │   ├── __pycache__
│   │   │   │   ├── __init__.cpython-310.pyc
│   │   │   │   ├── ctpn.cpython-310.pyc
│   │   │   │   ├── ctpn_anchor.cpython-310.pyc
│   │   │   │   ├── ctpn_connector.cpython-310.pyc
│   │   │   │   └── detection_utils.cpython-310.pyc
│   │   │   ├── ctpn.py
│   │   │   ├── ctpn_anchor.py
│   │   │   ├── ctpn_connector.py
│   │   │   ├── ctpn_functions.py
│   │   │   └── detection_utils.py
│   │   └── text_recognition
│   │       ├── CRNN.py
│   │       ├── HENNET.py
│   │       ├── __init__.py
│   │       ├── __pycache__
│   │       │   ├── CRNN.cpython-310.pyc
│   │       │   ├── HENNET.cpython-310.pyc
│   │       │   ├── __init__.cpython-310.pyc
│   │       │   ├── attention.cpython-310.pyc
│   │       │   ├── crnn_label_converter.cpython-310.pyc
│   │       │   ├── crnn_utils.cpython-310.pyc
│   │       │   ├── decoder.cpython-310.pyc
│   │       │   ├── encoder.cpython-310.pyc
│   │       │   ├── hennet_label_converter.cpython-310.pyc
│   │       │   ├── mha_formula.cpython-310.pyc
│   │       │   ├── position.cpython-310.pyc
│   │       │   ├── resnet.cpython-310.pyc
│   │       │   └── transformer.cpython-310.pyc
│   │       ├── attention.py
│   │       ├── crnn_label_converter.py
│   │       ├── crnn_utils.py
│   │       ├── decoder.py
│   │       ├── encoder.py
│   │       ├── hennet_label_converter.py
│   │       ├── jamo_utils
│   │       │   ├── __init__.py
│   │       │   ├── __pycache__
│   │       │   │   ├── __init__.cpython-310.pyc
│   │       │   │   ├── jamo_merge.cpython-310.pyc
│   │       │   │   └── jamo_split.cpython-310.pyc
│   │       │   ├── jamo_merge.py
│   │       │   └── jamo_split.py
│   │       ├── mha_formula.py
│   │       ├── modules
│   │       │   ├── __pycache__
│   │       │   │   ├── feature_extraction.cpython-310.pyc
│   │       │   │   ├── prediction.cpython-310.pyc
│   │       │   │   ├── sequence_modeling.cpython-310.pyc
│   │       │   │   └── transformation.cpython-310.pyc
│   │       │   ├── feature_extraction.py
│   │       │   ├── prediction.py
│   │       │   ├── sequence_modeling.py
│   │       │   └── transformation.py
│   │       ├── position.py
│   │       ├── resnet.py
│   │       ├── transformer.py
│   │       └── words.txt
│   └── recognizer
│       ├── __init__.py
│       ├── __pycache__
│       │   ├── __init__.cpython-310.pyc
│       │   ├── label_converter.cpython-310.pyc
│       │   └── recognizer.cpython-310.pyc
│       ├── jamo_utils
│       │   ├── __init__.py
│       │   ├── __pycache__
│       │   │   ├── __init__.cpython-310.pyc
│       │   │   ├── jamo_merge.cpython-310.pyc
│       │   │   └── jamo_split.cpython-310.pyc
│       │   ├── jamo_merge.py
│       │   └── jamo_split.py
│       ├── label_converter.py
│       ├── model
│       │   ├── __pycache__
│       │   │   ├── decoder.cpython-310.pyc
│       │   │   ├── encoder.cpython-310.pyc
│       │   │   ├── hen_net.cpython-310.pyc
│       │   │   └── transformation.cpython-310.pyc
│       │   ├── decoder.py
│       │   ├── encoder.py
│       │   ├── hen_net.py
│       │   ├── modules
│       │   │   ├── __pycache__
│       │   │   │   ├── attention.cpython-310.pyc
│       │   │   │   ├── mha_formula.cpython-310.pyc
│       │   │   │   ├── position.cpython-310.pyc
│       │   │   │   ├── resnet.cpython-310.pyc
│       │   │   │   └── transformer.cpython-310.pyc
│       │   │   ├── attention.py
│       │   │   ├── mha_formula.py
│       │   │   ├── multi_head_attention.py
│       │   │   ├── position.py
│       │   │   ├── resnet.py
│       │   │   └── transformer.py
│       │   └── transformation.py
│       ├── recognizer.py
│       └── recognizer.yaml
└── test.py


```

