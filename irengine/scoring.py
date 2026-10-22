"""Cosine and pivoted scoring, and ranking with a fixed tie-break (steps 12, 14, 15).

    S_cos(q, d) = sum_t w(t,q) w(t,d) / (n_q * n_d)
    S_piv(q, d) = sum_t w(t,q) w(t,d) / (n_q * ((1 - s) p + s n_d))

A zero-norm document or a zero-norm query scores 0. Query terms outside the
collection vocabulary were already dropped by ``InvertedIndex.vectorize``.

Ranking (step 15):
  * Only documents with score > 0 are retrieved. A document scoring 0 shares
    no weighted term with the query, so the engine has no evidence for it.
  * Order is by decreasing score; ties are broken by ascending document ID.
  * Scores are compared after rounding to TIE_DECIMALS decimal places, so two
    scores that are mathematically equal but differ by floating-point noise
    still count as a tie. The reported score is never rounded.
"""

from typing import Optional

from .index import DocId, InvertedIndex, Vector, check_slope, vector_norm

TIE_DECIMALS = 12

Ranking = list[tuple[DocId, float]]


def dot_products(index: InvertedIndex, qvec: Vector) -> dict[int, float]:
    """Term-at-a-time accumulation of w(t,q) * w(t,d) over the postings lists.

    Query terms are visited in sorted order so the floating-point sum for a
    document is always computed the same way."""
    acc: dict[int, float] = {}
    for term in sorted(qvec):
        wq = qvec[term]
        for pos, wd in index.weighted_postings[term]:
            if wd > 0:
                acc[pos] = acc.get(pos, 0.0) + wq * wd
    return acc


def cosine_scores(index: InvertedIndex, qvec: Vector) -> list[float]:
    """Cosine score for every document position (0.0 where undefined or no overlap)."""
    scores = [0.0] * index.N
    nq = vector_norm(qvec)
    if nq == 0:
        return scores
    for pos, dot in dot_products(index, qvec).items():
        nd = index.norms[pos]
        if nd > 0:
            scores[pos] = dot / (nq * nd)
    return scores


def pivoted_scores(
    index: InvertedIndex, qvec: Vector, s: float, pivot: Optional[float] = None
) -> list[float]:
    """Pivoted cosine score for every document position.

    ``pivot`` defaults to the index's frozen pivot; it can be overridden so later
    experiments (e.g. edited documents) keep the clean-collection value."""
    check_slope(s)
    p = index.pivot if pivot is None else pivot
    scores = [0.0] * index.N
    nq = vector_norm(qvec)
    if nq == 0:
        return scores
    for pos, dot in dot_products(index, qvec).items():
        nd = index.norms[pos]
        if nd > 0:
            scores[pos] = dot / (nq * ((1.0 - s) * p + s * nd))
    return scores


def rank(index: InvertedIndex, scores: list[float]) -> Ranking:
    """(doc_id, score) for every document with score > 0, best first."""
    hits = [(index.doc_ids[pos], sc) for pos, sc in enumerate(scores) if sc > 0]
    hits.sort(key=lambda hit: (-round(hit[1], TIE_DECIMALS), hit[0]))
    return hits


def search(
    index: InvertedIndex, query: str, s: Optional[float] = None, k: Optional[int] = None
) -> Ranking:
    """Rank the collection for a query: cosine when ``s`` is None, pivoted otherwise."""
    qvec = index.vectorize(query)
    scores = cosine_scores(index, qvec) if s is None else pivoted_scores(index, qvec, s)
    ranking = rank(index, scores)
    return ranking if k is None else ranking[:k]
