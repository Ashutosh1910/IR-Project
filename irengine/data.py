"""Loading the article collection, queries and relevance labels.

File formats
------------
articles: articles.json (list of objects) or articles.csv, with at least
          document_id, title, text.
queries:  JSON list of {"query_id", "text", ...} or CSV with query_id,text columns.
qrels:    CSV with query_id,document_id,relevance. relevance > 0 means relevant;
          explicit 0 rows are allowed (the LLM labels every pair).
"""

import csv
import json
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


def _parse_id(value):
    """Document IDs are ints when they look like ints (so 9999 sorts before 10806)."""
    if isinstance(value, int):
        return value
    text = str(value).strip()
    return int(text) if text.lstrip("-").isdigit() else text


def _read_records(path: Path) -> list[dict]:
    if path.suffix.lower() == ".json":
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    csv.field_size_limit(2**31 - 1)
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_articles(path: PathLike) -> list[dict]:
    articles = _read_records(Path(path))
    for a in articles:
        a["document_id"] = _parse_id(a["document_id"])
    return articles


def index_documents(articles: list[dict], include_title: bool = True) -> list[tuple]:
    """(document_id, text to index). By default the title is indexed with the body,
    since relevance judges see both."""
    return [
        (a["document_id"], f"{a['title']}\n{a['text']}" if include_title else a["text"])
        for a in articles
    ]


def load_queries(path: PathLike) -> dict:
    """query_id -> query text, in file order."""
    queries = {}
    for row in _read_records(Path(path)):
        qid = str(row["query_id"]).strip()
        if qid in queries:
            raise ValueError(f"duplicate query id {qid!r} in {path}")
        queries[qid] = row["text"]
    return queries


def load_qrels(path: PathLike, known_doc_ids=None) -> dict:
    """query_id -> set of relevant document IDs.

    Every labelled query gets an entry, even when none of its labels is positive,
    so an all-zero query can be told apart from an unlabelled one."""
    qrels: dict = {}
    for row in _read_records(Path(path)):
        qid = str(row["query_id"]).strip()
        doc_id = _parse_id(row["document_id"])
        if known_doc_ids is not None and doc_id not in known_doc_ids:
            raise ValueError(f"qrels row for {qid!r} names unknown document {doc_id!r}")
        rel = int(float(row["relevance"]))
        bucket = qrels.setdefault(qid, set())
        if rel > 0:
            bucket.add(doc_id)
    return qrels
