from typing import Optional

from .index import DocId, InvertedIndex, Vector, check_slope, vector_norm

# Scores equal to 12 decimals count as ties
TIE_DECIMALS = 12

Ranking = list[tuple[DocId, float]]


def dot_products(index: InvertedIndex, qvec: Vector) -> dict[int, float]:
    acc: dict[int, float] = {}
    # Sorted order keeps the float sums reproducible
    for term in sorted(qvec):
        wq = qvec[term]
        for pos, wd in index.weighted_postings[term]:
            if wd > 0:
                acc[pos] = acc.get(pos, 0.0) + wq * wd
    return acc


def cosine_scores(index: InvertedIndex, qvec: Vector) -> list[float]:
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
    check_slope(s)
    p = index.pivot if pivot is None else pivot
    scores = [0.0] * index.N
    nq = vector_norm(qvec)
    if nq == 0:
        return scores
    for pos, dot in dot_products(index, qvec).items():
        nd = index.norms[pos]
        if nd > 0:
            # Cosine with n_d replaced by the pivoted factor
            scores[pos] = dot / (nq * ((1.0 - s) * p + s * nd))
    return scores


def rank(index: InvertedIndex, scores: list[float]) -> Ranking:
    # Only documents with a positive score are retrieved
    hits = [(index.doc_ids[pos], sc) for pos, sc in enumerate(scores) if sc > 0]
    # Higher score first; ties go to the lower document ID
    hits.sort(key=lambda hit: (-round(hit[1], TIE_DECIMALS), hit[0]))
    return hits


def search(
    index: InvertedIndex, query: str, s: Optional[float] = None, k: Optional[int] = None
) -> Ranking:
    qvec = index.vectorize(query)
    scores = cosine_scores(index, qvec) if s is None else pivoted_scores(index, qvec, s)
    ranking = rank(index, scores)
    return ranking if k is None else ranking[:k]
