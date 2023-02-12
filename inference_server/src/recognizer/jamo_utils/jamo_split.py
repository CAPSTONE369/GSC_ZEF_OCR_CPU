import os, sys, re, itertools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jamo_utils import check_hangul, is_hangul_syllable, get_jamo_type, \
    INITIAL, MEDIAL, FINAL, CHAR_INDICES, CHAR_LISTS


def split_syllable_char(c):
    check_hangul(c)
    if len(c) != 1:
        raise ValueError("Input string must have exactly one character")
    
    init, med, final = None, None, None
    if is_hangul_syllable(c):
        offset = ord(c) - 0xac00
        x = (offset - offset % 28) // 28
        init, med, final = x // 21, x % 21, offset % 28 ## 번호 지정
        if not final:
            final = None ## 종성이 없는 경우
        else:
            final -= 1
    else:
        pos = get_jamo_type(c) ## 받침에 들어가는 syllable 중에서 흔히 아는 28개의 자/모음에 속하지 않는 것이 있음
        if pos & INITIAL == INITIAL:
            pos = INITIAL
        elif pos & MEDIAL == MEDIAL:
            pos = MEDIAL
        elif pos & FINAL == FINAL:
            pos = FINAL
        idx = CHAR_INDICES[pos][c] ## 초성-중성-종성에 해당하는 grapheme중에서 몇번쨰인지
        if pos == INITIAL:
            init = idx
        elif pos == MEDIAL:
            med = idx
        elif pos == FINAL:
            final = idx

    return tuple(
        CHAR_LISTS[pos][idx] if idx is not None else None
        for pos, idx in
        zip([INITIAL, MEDIAL, FINAL], [init, med, final])
    )

def split_syllables(s, ignore_err=True, pad=' '):
    def try_split(c):
        try:
            return split_syllable_char(c)
        except:
            replace = re.sub('[ sA-Za-z0-9,.()]', '', c) ## 영어나 숫자나 공백 
            if replace == '':
                return (c,)
            if replace == u'\u2227':
                return (c,)
            else:
                return '[UNK]' ## 특수 문자

    s = map(try_split, s)

    if pad is not None:
        tuples = map(lambda x: tuple(pad if y is None else y for y in x), s)
    else:
        tuples = map(lambda x: filter(None, x), s)
  
    ## 한글은 자음과 모음을 분리하고 영어와 숫자는 그대로 보내 준다.
    # 한편, 특수 문자의 경우에는 그냥 '[UNK]' 토큰으로 예측하도록 한다.

    return "".join(itertools.chain(*tuples))
    
if __name__ == "__main__":
    c = str(input())
    print(split_syllables(c))