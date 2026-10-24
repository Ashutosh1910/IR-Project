from functools import lru_cache

_VOWELS = frozenset("aeiou")


def _is_consonant(word: str, i: int) -> bool:
    ch = word[i]
    if ch in _VOWELS:
        return False
    if ch == "y":
        return i == 0 or not _is_consonant(word, i - 1)
    return True


# Porter's m: number of VC sequences in the stem
def _measure(stem: str) -> int:
    m, prev_vowel = 0, False
    for i in range(len(stem)):
        cons = _is_consonant(stem, i)
        if cons and prev_vowel:
            m += 1
        prev_vowel = not cons
    return m


def _has_vowel(stem: str) -> bool:
    return any(not _is_consonant(stem, i) for i in range(len(stem)))


def _ends_double_consonant(stem: str) -> bool:
    return len(stem) >= 2 and stem[-1] == stem[-2] and _is_consonant(stem, len(stem) - 1)


def _ends_cvc(stem: str) -> bool:
    return (
        len(stem) >= 3
        and _is_consonant(stem, len(stem) - 3)
        and not _is_consonant(stem, len(stem) - 2)
        and _is_consonant(stem, len(stem) - 1)
        and stem[-1] not in "wxy"
    )


def _replace_longest(word: str, rules, condition) -> str:
    for suffix, replacement in rules:
        if word.endswith(suffix):
            stem = word[: len(word) - len(suffix)]
            return stem + replacement if condition(stem) else word
    return word


def _sorted(rules):
    return sorted(rules, key=lambda r: -len(r[0]))


# Suffix rules for steps 2-4, longest suffix first
_STEP2 = _sorted([
    ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
    ("izer", "ize"), ("abli", "able"), ("alli", "al"), ("entli", "ent"), ("eli", "e"),
    ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"), ("ator", "ate"),
    ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"), ("ousness", "ous"),
    ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"),
])
_STEP3 = _sorted([
    ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"), ("ical", "ic"),
    ("ful", ""), ("ness", ""),
])
_STEP4 = _sorted([(s, "") for s in (
    "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment", "ent",
    "ion", "ou", "ism", "ate", "iti", "ous", "ive", "ize",
)])


def _step1a(w: str) -> str:
    if w.endswith("sses"):
        return w[:-2]
    if w.endswith("ies"):
        return w[:-2]
    if w.endswith("ss"):
        return w
    if w.endswith("s"):
        return w[:-1]
    return w


def _step1b(w: str) -> str:
    if w.endswith("eed"):
        return w[:-1] if _measure(w[:-3]) > 0 else w
    for suffix in ("ed", "ing"):
        if w.endswith(suffix):
            stem = w[: -len(suffix)]
            if not _has_vowel(stem):
                return w
            return _step1b_tidy(stem)
    return w


def _step1b_tidy(w: str) -> str:
    if w.endswith(("at", "bl", "iz")):
        return w + "e"
    if _ends_double_consonant(w) and w[-1] not in "lsz":
        return w[:-1]
    if _measure(w) == 1 and _ends_cvc(w):
        return w + "e"
    return w


def _step1c(w: str) -> str:
    if w.endswith("y") and _has_vowel(w[:-1]):
        return w[:-1] + "i"
    return w


def _step4(w: str) -> str:
    for suffix, _ in _STEP4:
        if w.endswith(suffix):
            stem = w[: -len(suffix)]
            ok = _measure(stem) > 1
            # -ion is removed only after s or t
            if suffix == "ion":
                ok = ok and stem.endswith(("s", "t"))
            return stem if ok else w
    return w


def _step5(w: str) -> str:
    if w.endswith("e"):
        stem = w[:-1]
        m = _measure(stem)
        if m > 1 or (m == 1 and not _ends_cvc(stem)):
            w = stem
    if w.endswith("ll") and _measure(w) > 1:
        w = w[:-1]
    return w


# Porter (1980) stemmer; words of one or two letters are kept
@lru_cache(maxsize=None)
def stem(word: str) -> str:
    if len(word) <= 2:
        return word
    w = _step1a(word)
    w = _step1b(w)
    w = _step1c(w)
    w = _replace_longest(w, _STEP2, lambda s: _measure(s) > 0)
    w = _replace_longest(w, _STEP3, lambda s: _measure(s) > 0)
    w = _step4(w)
    w = _step5(w)
    return w
