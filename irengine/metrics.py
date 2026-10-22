"""Evaluation metrics (step 16) and handling of queries without relevant documents (step 17).

For a ranking r_1, r_2, ... and relevant set R (|R| > 0):
    P@k  = |{r_1..r_k} ∩ R| / k          (always divided by k, even if fewer than k were retrieved)
    R@k  = |{r_1..r_k} ∩ R| / |R|
    F1@k = 2 P@k R@k / (P@k + R@k)       (0 when both are 0)
    AP   = (1/|R|) * sum over ranks i where r_i ∈ R of P@i
           -> relevant documents that were never retrieved contribute 0
    MAP  = mean AP over evaluated queries

Averages are macro averages over queries; mean F1@10 is the mean of per-query
F1@10 values (not the F1 of mean P and mean R). Queries with no relevant
documents are listed separately and excluded from every average.
"""

from dataclasses import dataclass, field
from typing import Hashable, Mapping, Optional, Sequence

QueryId = Hashable
DocId = Hashable


def _check(ranking: Sequence[DocId], k: Optional[int] = None) -> None:
    if k is not None and k <= 0:
        raise ValueError("k must be positive")
    if len(set(ranking)) != len(ranking):
        raise ValueError("ranking contains duplicate document ids")


def _hits_at(ranking: Sequence[DocId], relevant: set, k: int) -> int:
    return sum(1 for d in ranking[:k] if d in relevant)


def precision_at_k(ranking: Sequence[DocId], relevant: set, k: int = 10) -> float:
    _check(ranking, k)
    return _hits_at(ranking, relevant, k) / k


def recall_at_k(ranking: Sequence[DocId], relevant: set, k: int = 10) -> float:
    _check(ranking, k)
    if not relevant:
        raise ValueError("recall is undefined for a query with no relevant documents")
    return _hits_at(ranking, relevant, k) / len(relevant)


def f1_at_k(ranking: Sequence[DocId], relevant: set, k: int = 10) -> float:
    p = precision_at_k(ranking, relevant, k)
    r = recall_at_k(ranking, relevant, k)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def average_precision(ranking: Sequence[DocId], relevant: set) -> float:
    _check(ranking)
    if not relevant:
        raise ValueError("AP is undefined for a query with no relevant documents")
    hits, total = 0, 0.0
    for i, d in enumerate(ranking, start=1):
        if d in relevant:
            hits += 1
            total += hits / i
    return total / len(relevant)


@dataclass
class QueryResult:
    query_id: QueryId
    num_relevant: int
    num_retrieved: int
    precision: float
    recall: float
    f1: float
    average_precision: float


@dataclass
class NoRelevantQuery:
    query_id: QueryId
    num_retrieved: int
    labelled: bool  # False if the query had no relevance labels at all


@dataclass
class Evaluation:
    k: int
    per_query: list[QueryResult] = field(default_factory=list)
    no_relevant: list[NoRelevantQuery] = field(default_factory=list)

    def _mean(self, attr: str) -> Optional[float]:
        if not self.per_query:
            return None
        return sum(getattr(q, attr) for q in self.per_query) / len(self.per_query)

    @property
    def mean_precision(self):
        return self._mean("precision")

    @property
    def mean_recall(self):
        return self._mean("recall")

    @property
    def mean_f1(self):
        return self._mean("f1")

    @property
    def map(self):
        return self._mean("average_precision")


def evaluate(
    rankings: Mapping[QueryId, Sequence[DocId]],
    qrels: Mapping[QueryId, set],
    k: int = 10,
) -> Evaluation:
    """Evaluate one system's rankings.

    ``rankings`` decides which queries are evaluated (in its order). ``qrels`` maps a
    query to its set of relevant document IDs; a missing entry or an empty set
    puts the query in ``no_relevant`` instead of the averages."""
    result = Evaluation(k=k)
    for qid, ranking in rankings.items():
        relevant = qrels.get(qid)
        if not relevant:
            result.no_relevant.append(
                NoRelevantQuery(qid, len(ranking), labelled=qid in qrels)
            )
            continue
        result.per_query.append(
            QueryResult(
                query_id=qid,
                num_relevant=len(relevant),
                num_retrieved=len(ranking),
                precision=precision_at_k(ranking, relevant, k),
                recall=recall_at_k(ranking, relevant, k),
                f1=f1_at_k(ranking, relevant, k),
                average_precision=average_precision(ranking, relevant),
            )
        )
    return result
