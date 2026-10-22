"""Cranfield development collection: loading, binarization, query splits, slope choice (steps 19-23).

    relevance grades (ir_datasets "cranfield"):
       4  complete answer          3  high degree of relevance
       2  useful background/method 1  minimum interest (e.g. historical)
      -1  no interest
    a document is relevant  <=>  grade >= RELEVANCE_THRESHOLD (default 2)

Query splits (the single dev/held-out split and the cross-validation folds) are
seeded shuffles of all query IDs, made independently of the relevance labels, so
changing the threshold never changes a split.
"""

import random
from typing import Iterable, Mapping

from .data import _parse_id

DATASET = "cranfield"
RELEVANCE_THRESHOLD = 2
SPLIT_SEED = 15
DEV_FRACTION = 0.5
FOLDS = 5
SLOPES = [round(0.1 * i, 1) for i in range(11)]
SLOPE_TIE_DECIMALS = 12


def load_cranfield(dataset: str = DATASET):
    """(documents, queries, graded qrels) from ir_datasets.

    documents: [(doc_id, text)] in collection order. The ``text`` field already
               starts with the title, so the title is not added a second time.
    queries:   query_id -> text
    graded:    query_id -> {doc_id: grade}; every query has an entry.
    """
    import ir_datasets  # only the Cranfield experiments need it

    ds = ir_datasets.load(dataset)
    documents = [(_parse_id(d.doc_id), d.text) for d in ds.docs_iter()]
    queries = {q.query_id: q.text for q in ds.queries_iter()}
    graded: dict = {qid: {} for qid in queries}
    for r in ds.qrels_iter():
        if r.query_id not in graded:
            raise ValueError(f"qrels row for unknown query {r.query_id!r}")
        doc_id = _parse_id(r.doc_id)
        if doc_id in graded[r.query_id]:
            raise ValueError(f"duplicate qrels row for ({r.query_id!r}, {doc_id!r})")
        graded[r.query_id][doc_id] = r.relevance
    return documents, queries, graded


def binarize(graded: Mapping, threshold: int = RELEVANCE_THRESHOLD) -> dict:
    """query_id -> set of doc IDs with grade >= threshold (empty set if none)."""
    return {
        qid: {d for d, g in grades.items() if g >= threshold}
        for qid, grades in graded.items()
    }


def split_queries(
    query_ids: Iterable, seed: int = SPLIT_SEED, dev_fraction: float = DEV_FRACTION
) -> tuple[list, list]:
    """Seeded, fixed (dev, held-out) split; each part is returned in ID order.

    IDs are sorted before shuffling so the split does not depend on input order."""
    ids = sorted(query_ids, key=_parse_id)
    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)
    cut = round(len(ids) * dev_fraction)
    dev, held = set(shuffled[:cut]), set(shuffled[cut:])
    return [q for q in ids if q in dev], [q for q in ids if q in held]


def kfold_split(query_ids: Iterable, k: int = FOLDS, seed: int = SPLIT_SEED) -> list[list]:
    """Seeded, fixed partition into k folds of (nearly) equal size, each in ID order.

    Like ``split_queries`` it uses only the IDs, never the labels."""
    ids = sorted(query_ids, key=_parse_id)
    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)
    folds = [set(shuffled[i::k]) for i in range(k)]
    return [[q for q in ids if q in fold] for fold in folds]


def select_slope(map_by_slope: Mapping[float, float]) -> float:
    """The slope with the highest dev MAP; ties (to 12 decimals) go to the larger s,
    i.e. the choice closer to plain cosine."""
    return max(map_by_slope, key=lambda s: (round(map_by_slope[s], SLOPE_TIE_DECIMALS), s))
