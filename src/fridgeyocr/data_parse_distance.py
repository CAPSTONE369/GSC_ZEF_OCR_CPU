import re

def leave_han(a):
  a = re.sub(re.compile('[^가-힣]'), '', a)
  return a

def levenshtein_distance(a, b):
  """ Calculate the Levenshtein distance between two strings
  """
  # (1) 정규 표현식으로 오직 한글만 남겨둔다.
  a = leave_han(a)
  b = leave_han(b)
  # (2)
  if len(a) == 0:
    return len(b)
  if len(b) == 0:
    return len(a)
  if a[0] == b[0]:
    return levenshtein_distance(a[1:], b[1:])
  else:
    dist_1 = levenshtein_distance(a[1:], b)
    dist_2 = levenshtein_distance(a, b[1:])
    dist_3 = levenshtein_distance(a[1:], b[1:])
    return 1 + min([dist_1, dist_2, dist_3])


def levenshtein_score(a, b):
  a = leave_han(a)
  b = leave_han(b)

  dist = levenshtein_distance(a, b)
  score = ((len(a) + len(b)) - dist) / (len(a) + len(b))

  return score

ACCURATE_START_WORDS = ['수량', '단가', '단가수량']
ACCURATE_NAME_WORDS = ['상품명', '상품코드']
ACCURATE_STOP_WORDS = ['총품목수량', '합계', '총구매액', '과세', '부가세', '판매총액', '과세물품']

import heapq

def get_best_match(word_list, word):
  MAX_SCORE = -1
  MAX_WORD = ''

  for wl in word_list:
    score = levenshtein_score(wl, word)
    if score > MAX_SCORE:
      MAX_SCORE = score
      MAX_WORD = wl
  return -MAX_SCORE, MAX_WORD

def draw_roi(image, left_box, right_top):
  drawn = image.copy()
  cv2.rectangle(drawn, left_box, right_top, 255, 3)
  return drawn

def get_roi_of_recipt(han_dict, image):
  H, W, C = image.shape
  STOP = [];START = [];NAME = []
  for idx, info in enumerate(han_dict):
    name = info['name']
    box = info['box']
    heapq.heappush(STOP, [*get_best_match(ACCURATE_STOP_WORDS, name), idx])
    heapq.heappush(START, [*get_best_match(ACCURATE_START_WORDS, name), idx])
    heapq.heappush(NAME, [*get_best_match(ACCURATE_NAME_WORDS, name), idx])
  top_stop = heapq.heappop(STOP)
  top_start = heapq.heappop(START)
  top_name = heapq.heappop(NAME)
  print(f"STOP: {top_stop[1]} START: {top_start[1]} NAME: {top_name[1]}")

  right_top = han_dict[top_start[2]]['box']
  print(right_top)
  left_bot = han_dict[top_stop[2]]['box']

  left_bot = [0, left_bot[1]]
  right_top = [W, right_top[3]]
  drawn = draw_roi(image, left_bot, right_top)
  print(f"LEFT: {left_bot} RIGHT: {right_top}")

  return left_bot, right_top, drawn
  # answer = get_name_in_roi(left_bot, right_top, han_dict)

  # return answer



