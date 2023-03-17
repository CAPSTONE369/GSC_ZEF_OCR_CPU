import torch
import math
import cv2
import numpy as np
from torchvision import transforms

def normalize_pad(image, imgH, imgW):
    image = transforms.ToTensor()(image)
    image.sub_(0.5).div_(0.5)
    c, h, w = image.size()
    Pad_img = torch.FloatTensor(c, imgH, imgW).fill_(0)

    Pad_img[:, :, :w] = image
    if imgW != w:
        Pad_img[:, :, w:] = image[:, :, w-1].unsqueeze(2).expand(c, h, imgW-w)
    return Pad_img
    

def get_resize(image, imgH, imgW):
    h, w = image.shape
    ratio = w / float(h)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = math.ceil(imgH * ratio)
    resized = cv2.resize(image, (resized_w, imgH))
    return resized, resized_w

def change_to_tensor(image, imgH, imgW):
    # _, image = cv2.threshold(image, 127, 255, cv2.THRESH_OTSU)
    # image = cv2.equalizeHist(image)
    # image[np.where(image > 200)] = 255

    resized, resized_w = get_resize(image, imgH, imgW)
    tensor_image = normalize_pad(resized, imgH, imgW)
    return tensor_image



def get_crnn_output(model, tensor_image, prediction_mode, converter):
    model.eval()
    device = model.get_device()
    if len(tensor_image.shape) == 3:
        tensor_image = tensor_image.unsqueeze(0)
    tensor_image = tensor_image.to(device)
    with torch.no_grad():
        # for max length prediction
        length_for_pred = torch.IntTensor([25]).to(device)
        text_for_pred = torch.LongTensor(1, 25 + 1).fill_(0).to(device)
        if 'CTC' in prediction_mode:
            preds = model(tensor_image, text_for_pred)
            preds_size = torch.IntTensor([preds.size(1)])
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, preds_size)
        else:
            preds = model(tensor_image, text_for_pred, is_train=False)
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)
    if 'ATTN' in prediction_mode:
        pred_EOS = preds_str.find('[s]')
        preds_str = preds_str[:pred_EOS]
    
    return preds_str[0]

if __name__ == "__main__":
    image = cv2.imread('/home/guest/speaking_fridgey/ocr_exp_v3/deep-text-recognition-benchmark/demo/recipt_0_croped/3.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tensor_image = change_to_tensor(image, 32, 128)
    print(image.shape, tensor_image.shape)

