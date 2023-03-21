START_WORDS = ['수량', '단가', '단가수량', '금액']
STOP_WORDS = ['총품목수량', '결제', '과세', '부가세', '과세물품', '합계', '합', '면세물품', '총액', '총구매액', '할인금액', '활인금액', '신용카드', '면세', '카드', '판매총액']
NAME_WORDS = ['상품명', '품명', '상품코드', '코드',] # '상품']

## 불용어는 START WORDS, 
NOT_WORDS = STOP_WORDS + NAME_WORDS + ['할인',  '금액', '과세', '면세', '부가새', '품목', '할인', '에누리', '애누리', '특가', '세일', '릴레이', '물품', '수량'] # '총품목',]
import cv2

def match_string(source, target):
  if (source == target): # or (target in source):
    return True
  return False

def check_string(source, target):
  if (source in target) or (target in source):
    return True
  return False

def get_center(box):
  x1, y1, x2, y2 = box
  cx = (x1 + x2) // 2
  cy = (y1 + y2) // 2
  return [cx, cy]

def get_roi_point(box, pos):
  x1, y1, x2, y2 = box
  if pos.upper() == "R":
    px = x2
    py = y2
  else:
    px = 0
    py = y1
  return [px, py]
    
def draw_box(image, pt1, pt2):
  copied = image.copy()
  cv2.rectangle(copied, pt1, pt2, (255, 0, 0), thickness=3)

  return copied

def get_roi_with_keywords(han_dict, image):
  H,W,C=image.shape
  START = [];STOP = [];NAME = [];
  for info in han_dict:
    name = info['name']
    box = info['box']
  
    for target in START_WORDS:
      if match_string(name, target):
        if START == []:
          START = box
        else:
          if (box[1] > START[1]):
            START = box
    for target in STOP_WORDS:
      if match_string(name, target):
        if STOP == []:
          STOP = box
        else:
          if (box[1] < STOP[1]): # and (box[0] < STOP[0]):
            STOP = box

    for target in NAME_WORDS:
      if match_string(name, target):
        NAME = box

  print(START, STOP, NAME)
  if NAME != []:
    right_top = [W, NAME[3]]
    
  elif (START == STOP):
    right_top = [W, NAME[3]]
  elif (START == [] and NAME == []):
    righ_top = [W, 0]
  else:
    right_top = get_roi_point(START, "R")
    right_top[0] = max(W, right_top[0])
  if STOP == []:
    left_bot = [0, H]
  else:
    left_bot = get_roi_point(STOP, "L")


  return left_bot, right_top, draw_box(image, left_bot, right_top)

def vertical_nms(answer_dict):
  new_answer_dict = []
  Y = []
  for idx, info in enumerate(answer_dict):
    box = info['box']
    # print(info['name'])
    Y.append((box[1], box[3], idx, info["name"]))

  Y = sorted(Y, key = lambda x: (x[0], x[1]))

  used = [False for _ in range(len(Y))]
  for i in range(len(Y)-1):
    source = Y[i]
    source_h = (source[1] - source[0])
    # print(source[0], source[1])
    if used[i] == True:
      continue
    for j in range(i+1, i+2):
      target = Y[j]
      target_h = (target[1] - target[0])

      if abs((source_h//2 + source[0]) - (target_h//2 + target[0])) <= 5:

      # inter = min(target_h, target_w)

      # uni = max(target[1], source[1]) - min(target[0], source[0])
      # inter = min(target[1], source[1]) - max(target[0], source[0])
      # if (inter / uni) > 0.9:
        new_answer_dict.append({
            "name": source[-1] + target[-1]
        })
        used[i] = True
        used[j] = True
        break
    if used[i] == True:continue
    new_answer_dict.append({"name": source[-1]})
  if used[-1] == False:
    new_answer_dict.append({"name": Y[-1][-1]})
  return new_answer_dict






def check_in_roi(box, lx, ly, rx, ry):
  x1, y1, x2, y2 = box
  if (lx <= x1 <= rx) and (ry < y1 < ly) and (lx <= x2 <= rx) and (ry < y2 < ly):
    return True
  return False

def get_name_in_roi(left_bot, right_top, han_dict):
  lx, ly = left_bot;rx, ry = right_top;
  answer = []
  for info in han_dict:
    name = info['name']
    box = info['box']
    not_in=True
    if check_in_roi(box, lx, ly, rx, ry):
      for target in NOT_WORDS:
        if check_string(name, target) == True:
          not_in = False
    
      if not_in == True:
        answer.append({"name": name, "box": box})
        
  answer = vertical_nms(answer)
  print(answer)
  return answer


