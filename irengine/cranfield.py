import random
from typing import Iterable, Mapping

from .data import _parse_id

DATASET = "cranfield"
# Grades 2-4 count as relevant
RELEVANCE_THRESHOLD = 2
SPLIT_SEED = 15
DEV_FRACTION = 0.5
FOLDS = 5
SLOPES = [round(0.1 * i, 1) for i in range(11)]
SLOPE_TIE_DECIMALS = 12


def load_cranfield(dataset: str = DATASET):
    # Imported here so only Cranfield runs need it
    import ir_datasets

    ds = ir_datasets.load(dataset)
    # The text field already starts with the title
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
    return {
        qid: {d for d, g in grades.items() if g >= threshold}
        for qid, grades in graded.items()
    }


# Seeded dev/held-out split of query IDs, independent of labels
def split_queries(
    query_ids: Iterable, seed: int = SPLIT_SEED, dev_fraction: float = DEV_FRACTION
) -> tuple[list, list]:
    ids = sorted(query_ids, key=_parse_id)
    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)
    cut = round(len(ids) * dev_fraction)
    dev, held = set(shuffled[:cut]), set(shuffled[cut:])
    return [q for q in ids if q in dev], [q for q in ids if q in held]


# Seeded k-fold split of query IDs
def kfold_split(query_ids: Iterable, k: int = FOLDS, seed: int = SPLIT_SEED) -> list[list]:
    ids = sorted(query_ids, key=_parse_id)
    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)
    folds = [set(shuffled[i::k]) for i in range(k)]
    return [[q for q in ids if q in fold] for fold in folds]


# Highest MAP wins; ties go to the larger s
def select_slope(map_by_slope: Mapping[float, float]) -> float:
    return max(map_by_slope, key=lambda s: (round(map_by_slope[s], SLOPE_TIE_DECIMALS), s))
