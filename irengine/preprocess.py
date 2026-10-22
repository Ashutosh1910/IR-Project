"""Text preprocessing shared by documents and queries (step 10).

Pipeline, applied identically to articles and queries:
  1. Unicode NFKD, then drop accents ("café" -> "cafe") and invisible characters
     (soft hyphen, zero-width spaces, word joiner, BOM).
  2. Map curly/modifier apostrophes to "'" and lowercase.
  3. Drop possessive "'s" ("EC's" -> "ec"), then delete remaining apostrophes so
     contractions stay one token ("don't" -> "dont").
  4. Tokens are maximal runs of [a-z0-9]; everything else separates tokens
     (so "cool-down" -> "cool", "down").
  5. Remove the fixed stopword list. No stemming.
"""

import re
import unicodedata

from .stopwords import RAW_STOPWORDS

_DROP_CHARS = dict.fromkeys(map(ord, "­​‌‍⁠﻿"))
_APOSTROPHES = str.maketrans({c: "'" for c in "‘’ʼ′`"})
_POSSESSIVE = re.compile(r"'s\b")
_TOKEN = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    """Steps 1-3: canonical lowercase text with apostrophes resolved."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.translate(_DROP_CHARS).translate(_APOSTROPHES).lower()
    text = _POSSESSIVE.sub("", text)
    return text.replace("'", "")


def tokenize(text: str) -> list[str]:
    """Steps 1-4: tokens before stopword removal."""
    return _TOKEN.findall(normalize(text))


STOPWORDS = frozenset(tok for word in RAW_STOPWORDS for tok in tokenize(word))


def preprocess(text: str) -> list[str]:
    """Full pipeline: the token list that gets indexed or used as a query."""
    return [tok for tok in tokenize(text) if tok not in STOPWORDS]
