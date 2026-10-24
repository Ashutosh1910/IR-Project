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
    def __init__(
        self,
        documents: Iterable[tuple[DocId, str]],
        preprocess: Callable[[str], list[str]] = default_preprocess,
    ):
        self.preprocess = preprocess
        self.doc_ids: list[DocId] = []
        self.doc_lengths: list[int] = []
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

        self.postings = {t: raw_postings[t] for t in sorted(raw_postings)}
        self.df = {t: len(plist) for t, plist in self.postings.items()}
        # IDF is used only in query weights (lnc.ltc)
        self.idf = {t: math.log(self.N / df) for t, df in self.df.items()}

        # Document weights: 1 + ln tf, no IDF
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
        # Pivot: mean norm over nonzero-norm documents
        self.pivot = math.fsum(nonzero) / len(nonzero) if nonzero else 0.0

    @property
    def vocabulary_size(self) -> int:
        return len(self.postings)

    def weigh_counts(self, counts: dict[str, int]) -> Vector:
        vec: Vector = {}
        for term, count in counts.items():
            idf = self.idf.get(term)
            if idf is None or count <= 0:
                continue
            # Query weight: (1 + ln tf) * IDF; unseen terms skipped
            w = log_tf(count) * idf
            if w > 0:
                vec[term] = w
        return vec

    def vectorize(self, text: str) -> Vector:
        return self.weigh_counts(Counter(self.preprocess(text)))

    def document_vector(self, doc_id: DocId) -> Vector:
        pos = self.position[doc_id]
        return {
            t: w
            for t, plist in self.weighted_postings.items()
            for p, w in plist
            if p == pos and w > 0
        }

    def pivoted_factor(self, norm: float, s: float) -> float:
        check_slope(s)
        # Pivoted normalization factor
        return (1.0 - s) * self.pivot + s * norm

    def pivoted_factors(self, s: float) -> list[float]:
        return [self.pivoted_factor(n, s) if n > 0 else 0.0 for n in self.norms]


def check_slope(s: float) -> None:
    if not 0.0 <= s <= 1.0:
        raise ValueError(f"slope s must be in [0, 1], got {s}")
