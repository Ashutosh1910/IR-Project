import re
import unicodedata

from .porter import stem
from .stopwords import RAW_STOPWORDS

# Invisible characters removed before tokenizing
_DROP_CHARS = dict.fromkeys(map(ord, "­​‌‍⁠﻿"))
_APOSTROPHES = str.maketrans({c: "'" for c in "‘’ʼ′`"})
_POSSESSIVE = re.compile(r"'s\b")
_TOKEN = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.translate(_DROP_CHARS).translate(_APOSTROPHES).lower()
    # Drop possessive 's, then any remaining apostrophes
    text = _POSSESSIVE.sub("", text)
    return text.replace("'", "")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(normalize(text))


# Stopwords get the same normalisation as the text
STOPWORDS = frozenset(tok for word in RAW_STOPWORDS for tok in tokenize(word))


def preprocess(text: str) -> list[str]:
    # Remove stopwords, then Porter-stem
    return [stem(tok) for tok in tokenize(text) if tok not in STOPWORDS]
