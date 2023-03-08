__all__ = ["jamo_merge", "jamo_split",       ## 음소들을 합쳐주고 이를 바탕으로 단어 만들기
            "CHAR_LISTS", "CHAR_INITIALS", "CHAR_MEDIALS", "CHAR_FINALS"] ## 초성, 중성, 종성

import itertools
INITIAL= 0x001
MEDIAL = 0x010
FINAL = 0x100

CHAR_LISTS = {
    INITIAL: list(map(chr, [
        0x3131, 0x3132, 0x3134, 0x3137, 0x3138, 0x3139,
        0x3141, 0x3142, 0x3143, 0x3145, 0x3146, 0x3147,
        0x3148, 0x3149, 0x314a, 0x314b, 0x314c, 0x314d,
        0x314e
    ])), ## ㄱ -> ㅎ 까지를 아우르는 (초성)
    MEDIAL: list(map(chr, [
        0x314f, 0x3150, 0x3151, 0x3152, 0x3153, 0x3154,
        0x3155, 0x3156, 0x3157, 0x3158, 0x3159, 0x315a,
        0x315b, 0x315c, 0x315d, 0x315e, 0x315f, 0x3160,
        0x3161, 0x3162, 0x3163
    ])), ## ㅏ -> ㅣ 까지를 아우르는 (중성)
    FINAL: list(map(chr, [
        0x3131, 0x3132, 0x3133, 0x3134, 0x3135, 0x3136,
        0x3137, 0x3139, 0x313a, 0x313b, 0x313c, 0x313d,
        0x313e, 0x313f, 0x3140, 0x3141, 0x3142, 0x3144,
        0x3145, 0x3146, 0x3147, 0x3148, 0x314a, 0x314b,
        0x314c, 0x314d, 0x314e
    ])) ## final of the grapheme (종성)
}

CHAR_INITIALS=CHAR_LISTS[INITIAL]
CHAR_MEDIALS=CHAR_LISTS[MEDIAL]
CHAR_FINALS=CHAR_LISTS[FINAL]

CHAR_SETS={k:set(v) for k, v in CHAR_LISTS.items()} ## 초성과 종성에는 중복이 되는 문자도 존재하기 때문이다.
CHARSET = set(itertools.chain(*CHAR_SETS.values()))
CHAR_INDICES = {k: {c: i for i, c in enumerate(v)} \
    for k, v in CHAR_LISTS.items()}

def is_hangul_syllable(c):
    return 0xac00 <= ord(c) <= 0xd7a3 ## 한글 syllable에 속하는지 확인

def is_hangul_jamo(c):
    return 0x1100 <= ord(c) <= 0x11ff ## 한글 자음/모음에 속하는지 확인

def is_hangul_compat_jamo(c):
    return 0x3130 <= ord(c) <= 0x318f ## 한글 compatibility 자음/모음에 속하는지 확인

def is_hangul_jamo_exta(c):
    return 0xa960 <= ord(c) <= 0xa97f ## 한글 자음/모음 Extended-A에 속하는지 확인

def is_hangul_jamo_extb(c):
    return 0xd7b0 <= ord(c) <= 0xd7ff ## 한글 자음/모음 Extended-B에 속하는지 확인

def is_hangul(c):
    return (is_hangul_syllable(c) or \
            is_hangul_jamo(c) or \
            is_hangul_compat_jamo(c) or \
            is_hangul_jamo_exta(c) or \
            is_hangul_jamo_extb(c))

def is_supported_hangul(c):
    return is_hangul_syllable(c) or is_hangul_compat_jamo(c)


def check_hangul(c, jamo_only=False):
    if not ((jamo_only or is_hangul_compat_jamo(c)) or is_supported_hangul(c)):
        raise ValueError(f"'{c}' is not a supported hangul character. "
                         f"'Hangul Syllables' (0xac00 ~ 0xd7a3) and "
                         f"'Hangul Compatibility Jamos' (0x3130 ~ 0x318f) are "
                         f"supported at the moment.")

def get_jamo_type(c):
    check_hangul(c)
    assert is_hangul_compat_jamo(c), f"not a jamo: {ord(c):x}"
    return sum(t for t, s in CHAR_SETS.items() if c in s)