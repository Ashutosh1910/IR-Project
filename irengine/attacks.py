"""Text-manipulation attacks on one document at a time, with collection statistics frozen.

Three attacks are appended to the end of a target article (the engine is bag-of-words,
so position has no effect):

  repeat    the query's terms (its words after stopword removal, unstemmed) k times
  pad       unrelated text equal to a percentage of the article's body word count
  combined  repetition first, then padding; the padding % is of the original length

Only the edited document's weights 1 + ln tf and its norm are recomputed. The index is
never modified: IDF, N and the pivot p keep their clean-collection values, and query
vectors come from the clean index. Under lnc.ltc documents carry no IDF, so the frozen
IDF only affects query weights.

Targets are chosen once and used for both methods: for each query, non-relevant
articles outside the top 10 under both cosine and pivoted, ``per_group`` from each
length group, sampled with a fixed seed.
"""

import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from .index import DocId, InvertedIndex, Vector, check_slope, log_tf, vector_norm
from .preprocess import STOPWORDS, preprocess, tokenize
from .scoring import Ranking, cosine_scores, pivoted_scores, rank

SEED = 15
K = 10
REPETITIONS = (5, 10, 25, 50)
PADDING_PCTS = (25, 50, 100, 200)
GROUPS = ("short", "medium", "long")

_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n+")


# ---------------------------------------------------------------- attack levels

@dataclass(frozen=True)
class Attack:
    kind: str          # "repeat", "pad" or "combined"
    repeats: int = 0   # k, times the query terms are appended
    pad_pct: int = 0   # padding as % of the original body word count

    @property
    def label(self) -> str:
        if self.kind == "repeat":
            return f"repeat x{self.repeats}"
        if self.kind == "pad":
            return f"pad {self.pad_pct}%"
        return f"repeat x{self.repeats} + pad {self.pad_pct}%"


def attack_grid() -> list[Attack]:
    """4 repetition levels, 4 padding levels and the full 4 x 4 combined grid (24 attacks)."""
    return (
        [Attack("repeat", repeats=k) for k in REPETITIONS]
        + [Attack("pad", pad_pct=p) for p in PADDING_PCTS]
        + [Attack("combined", repeats=k, pad_pct=p) for k in REPETITIONS for p in PADDING_PCTS]
    )


# ---------------------------------------------------------------- targets

def top_k(ranking: Ranking, k: int = K) -> set:
    return {d for d, _ in ranking[:k]}


def select_targets(
    index: InvertedIndex,
    queries: Mapping[str, str],
    qrels: Mapping[str, set],
    groups: Mapping[DocId, str],
    s: float,
    per_group: int = 2,
    seed: int = SEED,
    k: int = K,
) -> dict[str, list[DocId]]:
    """query_id -> target document IDs, ``per_group`` from each length group.

    Eligible: not relevant for the query, and outside the top k under both cosine
    and pivoted (unretrieved documents count as outside). One generator, seeded
    once, is used over the queries in file order, so the selection is reproducible."""
    rng = random.Random(seed)
    targets: dict[str, list[DocId]] = {}
    for qid, text in queries.items():
        qvec = index.vectorize(text)
        top = top_k(rank(index, cosine_scores(index, qvec)), k) | top_k(
            rank(index, pivoted_scores(index, qvec, s)), k)
        chosen: list[DocId] = []
        for g in GROUPS:
            pool = sorted(d for d in index.doc_ids
                          if groups[d] == g and d not in qrels[qid] and d not in top)
            if len(pool) < per_group:
                raise ValueError(f"{qid}: only {len(pool)} eligible {g} articles")
            chosen.extend(rng.sample(pool, per_group))
        targets[qid] = chosen
    return targets


# ---------------------------------------------------------------- added text

def query_terms(query_text: str) -> list[str]:
    """The query's words after stopword removal, unstemmed, in query order.

    Repeated words stay repeated (Q9 has "sports" twice), matching the query vector."""
    return [tok for tok in tokenize(query_text) if tok not in STOPWORDS]


def repetition_text(query_text: str, repeats: int) -> str:
    return " ".join(query_terms(query_text) * repeats)


@dataclass(frozen=True)
class Sentence:
    doc_id: DocId
    text: str
    stems: frozenset


def split_sentences(articles: Sequence[dict]) -> list[Sentence]:
    """Body sentences of every article, in document-ID order, with their stems
    (computed once and reused for every query and target)."""
    out = []
    for a in sorted(articles, key=lambda a: a["document_id"]):
        for sent in _SENTENCE_BREAK.split(a["text"]):
            if sent.strip():
                out.append(Sentence(a["document_id"], sent, frozenset(preprocess(sent))))
    return out


def padding_stream(
    sentences: Sequence[Sentence],
    query_text: str,
    relevant: set,
    target: DocId,
    seed: int = SEED,
    qid: str = "",
) -> list[str]:
    """Shuffled words for padding one target for one query.

    Source: body sentences of the other articles that are not relevant for the query.
    Any sentence containing a query stem (after full preprocessing) is dropped, so the
    padding cannot add a query term or answer the query. The sentence order is shuffled
    with a generator seeded by (seed, query, target), so every padding level for a
    target is a prefix of the same stream and the combined attack reuses the same text."""
    q_stems = set(preprocess(query_text))
    pool = [x.text for x in sentences
            if x.doc_id != target and x.doc_id not in relevant and not q_stems & x.stems]
    random.Random(f"{seed}:{qid}:{target}").shuffle(pool)
    return [w for sent in pool for w in sent.split()]


def padding_words(word_count: int, pct: int) -> int:
    """Words of padding for pct % of the body word count, rounded half up."""
    return math.floor(word_count * pct / 100 + 0.5)


def padding_text(stream: Sequence[str], n_words: int, query_text: str) -> str:
    """The first n_words words of the stream, checked to contain no query stem."""
    if n_words > len(stream):
        raise ValueError(f"padding pool has {len(stream)} words, {n_words} needed")
    text = " ".join(stream[:n_words])
    overlap = set(preprocess(query_text)) & set(preprocess(text))
    assert not overlap, f"padding contains query stems {sorted(overlap)}"
    return text


def added_text(attack: Attack, query_text: str, stream: Sequence[str], word_count: int) -> str:
    parts = []
    if attack.repeats:
        parts.append(repetition_text(query_text, attack.repeats))
    if attack.pad_pct:
        parts.append(padding_text(stream, padding_words(word_count, attack.pad_pct), query_text))
    return " ".join(parts)


# ---------------------------------------------------------------- edited document

@dataclass
class EditedDocument:
    weights: Vector       # w(t, d) = 1 + ln tf over the edited text
    norm: float
    added_words: int      # whitespace-separated words appended
    added_tokens: int     # tokens indexed from them (after stopwords)


def edited_document(base_counts: Counter, extra: str) -> EditedDocument:
    """Weights and norm of a document after appending ``extra``.

    ``base_counts`` are the original document's term counts (title + body, as indexed).
    Appending is done on counts: the original text and the extra text are separated
    by a newline, which no token spans, so this equals preprocessing the joined text."""
    extra_tokens = preprocess(extra) if extra else []
    counts = base_counts + Counter(extra_tokens)
    weights = {t: log_tf(c) for t, c in counts.items()}
    return EditedDocument(weights, vector_norm(weights), len(extra.split()), len(extra_tokens))


def edited_score(
    index: InvertedIndex, qvec: Vector, doc: EditedDocument, s: Optional[float] = None
) -> float:
    """Score of the edited document: cosine when s is None, else pivoted with the
    frozen clean-collection pivot. Mirrors ``scoring.dot_products`` term order so an
    empty edit reproduces the clean score exactly."""
    nq = vector_norm(qvec)
    if nq == 0 or doc.norm == 0:
        return 0.0
    dot = 0.0
    for term in sorted(qvec):
        wd = doc.weights.get(term, 0.0)
        if wd > 0:
            dot += qvec[term] * wd
    if s is None:
        return dot / (nq * doc.norm)
    check_slope(s)
    return dot / (nq * ((1.0 - s) * index.pivot + s * doc.norm))


def attacked_ranking(
    index: InvertedIndex, clean_scores: Sequence[float], target: DocId, new_score: float
) -> Ranking:
    """Ranking with only the target's score replaced; all other scores stay clean."""
    scores = list(clean_scores)
    scores[index.position[target]] = new_score
    return rank(index, scores)


def rank_of(ranking: Ranking, doc_id: DocId, unretrieved: int) -> int:
    """1-based rank, or ``unretrieved`` (N + 1) when the document has score 0."""
    for i, (d, _) in enumerate(ranking, start=1):
        if d == doc_id:
            return i
    return unretrieved
