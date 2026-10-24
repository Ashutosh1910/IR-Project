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
    # Always divided by k, even if fewer were retrieved
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
    # Relevant documents never retrieved add 0
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
    labelled: bool


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
    result = Evaluation(k=k)
    for qid, ranking in rankings.items():
        # Queries without relevant documents are left out of averages
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
