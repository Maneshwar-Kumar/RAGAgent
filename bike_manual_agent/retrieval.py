"""Hybrid (semantic + keyword) search over ingested manuals."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import config

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = set(
    "a an and are as at be by for from has have i if in is it its my of on or so that the this to "
    "was what when where which why will with you your me do does did can how".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(d) for d in docs]
        self.lens = np.array([len(d) for d in docs], dtype=np.float32)
        self.avg = float(self.lens.mean()) if len(docs) else 0.0
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.tfs), dtype=np.float32)
        for i, tf in enumerate(self.tfs):
            norm = self.k1 * (1 - self.b + self.b * self.lens[i] / (self.avg or 1))
            out[i] = sum(
                self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in query if t in tf
            )
        return out


@dataclass
class Manual:
    manual_id: str
    meta: dict
    chunks: list[dict]
    vectors: np.ndarray
    bm25: BM25

    @classmethod
    def load(cls, folder: Path) -> "Manual":
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        chunks = json.loads((folder / "chunks.json").read_text(encoding="utf-8"))
        vectors = np.load(folder / "embeddings.npy")
        bm25 = BM25([tokenize(f"{c['section'] or ''} {c['text']}") for c in chunks])
        return cls(meta["manual_id"], meta, chunks, vectors, bm25)


def list_manual_ids(index_dir: Path | None = None) -> list[str]:
    root = index_dir or config.INDEX_DIR
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "meta.json").exists())


@lru_cache(maxsize=16)
def _load_cached(folder: str, mtime: float) -> Manual:
    return Manual.load(Path(folder))


def load_manual(manual_id: str, index_dir: Path | None = None) -> Manual:
    folder = (index_dir or config.INDEX_DIR) / manual_id
    if not (folder / "meta.json").exists():
        raise KeyError(manual_id)
    return _load_cached(str(folder), (folder / "embeddings.npy").stat().st_mtime)


def embed_query(text: str) -> np.ndarray:
    """Embed a search query. Cached by its lower-cased, whitespace-normalised text, because
    agents often repeat searches such as "troubleshooting" in slightly different forms."""
    return _embed_normalised(" ".join(text.lower().split()))


@lru_cache(maxsize=2048)
def _embed_normalised(text: str) -> np.ndarray:
    from .ingest import embed_texts

    vector = embed_texts([text], "RETRIEVAL_QUERY")[0]
    vector.setflags(write=False)
    return vector


embed_query.cache_clear = _embed_normalised.cache_clear


def citation(chunk: dict) -> str:
    page = f"p. {chunk['printed_page']}" if chunk["printed_page"] else f"PDF p. {chunk['pdf_page']}"
    if chunk["printed_page"]:
        page += f" (PDF p. {chunk['pdf_page']})"
    return f"{page}{' - ' + chunk['section'] if chunk['section'] else ''}"


def cite_label(chunk: dict) -> str:
    """Short bracketed label the agent must use when citing this chunk."""
    if chunk["printed_page"]:
        return f"[p. {chunk['printed_page']}]"
    return f"[PDF p. {chunk['pdf_page']}]"


def search(manual: Manual, query: str, top_k: int | None = None, query_vector=None) -> dict:
    """Return the top chunks by reciprocal-rank fusion of cosine and BM25 rankings."""
    top_k = top_k or config.TOP_K
    qv = embed_query(query) if query_vector is None else query_vector
    cosine = manual.vectors @ qv
    keyword = manual.bm25.scores(tokenize(query))

    rrf = np.zeros(len(manual.chunks), dtype=np.float32)
    keyword_hits = [i for i in np.argsort(-keyword) if keyword[i] > 0]  # no overlap, no credit
    for ranking in (np.argsort(-cosine), keyword_hits):
        for rank, idx in enumerate(ranking[:50]):
            rrf[idx] += 1.0 / (60 + rank)
    best = np.argsort(-rrf)[:top_k]
    results = []
    for idx in best:
        c = manual.chunks[int(idx)]
        results.append(
            {
                "chunk_id": c["chunk_id"],
                "cite_as": cite_label(c),
                "location": citation(c),
                "pdf_page": c["pdf_page"],
                "printed_page": c["printed_page"],
                "section": c["section"],
                "similarity": round(float(cosine[idx]), 3),
                "text": c["text"],
            }
        )
    return {"manual": manual.meta["name"], "query": query, "results": results}
