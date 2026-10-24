"""Inverted index with the paper's lnc.ltc weights, vector norms and the pivot (steps 10, 11, 13).

Weights follow Singhal, Buckley & Mitra (1996), section 3: "the idf factor is only
used in the query term weights and not in the document term weights" (SMART lnc.ltc).

    TF(t, x)  = 1 + ln f(t, x)   if f(t, x) > 0, else 0
    IDF(t)    = ln(N / df(t))
    w(t, d)   = TF(t, d)                document weight (l n c: log TF, no IDF)
    w(t, q)   = TF(t, q) * IDF(t)       query weight    (l t c: log TF, IDF)
    n_d       = sqrt(sum_t w(t, d)^2)
    p         = mean of n_d over documents with n_d > 0

Query terms that are not in the collection vocabulary are ignored. A term found in
every document has IDF 0, so it gets query weight 0 and never adds to a score, but it
still counts toward the document norms, as in the paper.
"""

import math
from collections import Counter
from typing import Callable, Hashable, Iterable

from .preprocess import preprocess as default_preprocess

DocId = Hashable
Vector = dict[str, float]


def log_tf(count: int) -> float:
    return 1.0 + math.log(count) if count > 0 else 0.0


def vector_norm(vec: Vector) -> float:
    return math.sqrt(math.fsum(w * w for w in vec.values()))


class InvertedIndex:
    """Built once from a fixed collection; IDF and the pivot are frozen afterwards.

    Documents are addressed internally by position (0..N-1) in the order given;
    ``doc_ids[i]`` is the external ID used for output and tie-breaking.
    """

    def __init__(
        self,
        documents: Iterable[tuple[DocId, str]],
        preprocess: Callable[[str], list[str]] = default_preprocess,
    ):
        self.preprocess = preprocess
        self.doc_ids: list[DocId] = []
        self.doc_lengths: list[int] = []  # tokens after preprocessing
        raw_postings: dict[str, list[tuple[int, int]]] = {}
        seen: set = set()

        for doc_id, text in documents:
            if doc_id in seen:
                raise ValueError(f"duplicate document id: {doc_id!r}")
            seen.add(doc_id)
            pos = len(self.doc_ids)
            self.doc_ids.append(doc_id)
            tokens = preprocess(text)
            self.doc_lengths.append(len(tokens))
            for term, count in Counter(tokens).items():
                raw_postings.setdefault(term, []).append((pos, count))

        self.N = len(self.doc_ids)
        self.position = {doc_id: pos for pos, doc_id in enumerate(self.doc_ids)}

        # term -> [(doc position, raw count)], positions ascending
        self.postings = {t: raw_postings[t] for t in sorted(raw_postings)}
        self.df = {t: len(plist) for t, plist in self.postings.items()}
        self.idf = {t: math.log(self.N / df) for t, df in self.df.items()}

        # term -> [(doc position, w(t, d))]; used for scoring
        self.weighted_postings = {
            t: [(pos, log_tf(count)) for pos, count in plist]
            for t, plist in self.postings.items()
        }

        squares: list[list[float]] = [[] for _ in range(self.N)]
        for plist in self.weighted_postings.values():
            for pos, w in plist:
                squares[pos].append(w * w)
        self.norms = [math.sqrt(math.fsum(sq)) for sq in squares]

        nonzero = [n for n in self.norms if n > 0]
        self.num_nonzero_docs = len(nonzero)
        # Pivot over nonzero-norm documents only (step 13). With no such
        # document every score is 0 anyway, so the value 0.0 is never used.
        self.pivot = math.fsum(nonzero) / len(nonzero) if nonzero else 0.0

    @property
    def vocabulary_size(self) -> int:
        return len(self.postings)

    def weigh_counts(self, counts: dict[str, int]) -> Vector:
        """Query weights w(t, q) = TF(t, q) * IDF(t) for raw term counts.

        Terms outside the vocabulary are dropped, as are zero weights
        (a term in every document has IDF 0)."""
        vec: Vector = {}
        for term, count in counts.items():
            idf = self.idf.get(term)
            if idf is None or count <= 0:
                continue
            w = log_tf(count) * idf
            if w > 0:
                vec[term] = w
        return vec

    def vectorize(self, text: str) -> Vector:
        """Preprocess a query and weight it against this collection."""
        return self.weigh_counts(Counter(self.preprocess(text)))

    def document_vector(self, doc_id: DocId) -> Vector:
        """The stored weight vector of an indexed document (slow; for inspection/tests)."""
        pos = self.position[doc_id]
        return {
            t: w
            for t, plist in self.weighted_postings.items()
            for p, w in plist
            if p == pos and w > 0
        }

    def pivoted_factor(self, norm: float, s: float) -> float:
        """n'_d = (1 - s) p + s n_d (step 14)."""
        check_slope(s)
        return (1.0 - s) * self.pivot + s * norm

    def pivoted_factors(self, s: float) -> list[float]:
        """n'_d for every document; zero-norm documents get 0.0 (they always score 0)."""
        return [self.pivoted_factor(n, s) if n > 0 else 0.0 for n in self.norms]


def check_slope(s: float) -> None:
    if not 0.0 <= s <= 1.0:
        raise ValueError(f"slope s must be in [0, 1], got {s}")
