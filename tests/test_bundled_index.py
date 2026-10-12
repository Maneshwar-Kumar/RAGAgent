"""The committed manuals and their prebuilt index must stay in step (no model calls).

If a PDF, the chunking settings or the embedding model change, these tests fail until the
index is rebuilt with `python scripts/fetch_manuals.py --ingest`.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from bike_manual_agent import config, retrieval

ROOT = Path(__file__).resolve().parents[1]
SOURCES = json.loads((ROOT / "manuals" / "sources.json").read_text(encoding="utf-8"))["manuals"]


@pytest.mark.parametrize("entry", SOURCES, ids=[e["manual_id"] for e in SOURCES])
def test_index_was_built_from_the_committed_pdf(entry):
    pdf = ROOT / "manuals" / entry["file"]
    assert hashlib.sha256(pdf.read_bytes()).hexdigest() == entry["sha256"]

    folder = ROOT / "index" / entry["manual_id"]
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    assert meta["source_sha256"] == entry["sha256"]
    assert meta["embedding_model"] == config.EMBEDDING_MODEL
    assert (meta["chunk_chars"], meta["chunk_overlap"]) == (config.CHUNK_CHARS, config.CHUNK_OVERLAP)

    chunks = json.loads((folder / "chunks.json").read_text(encoding="utf-8"))
    vectors = np.load(folder / "embeddings.npy")
    assert len(chunks) == meta["chunks"] == vectors.shape[0]
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3)


def test_search_over_the_real_index_needs_no_model_call():
    manual = retrieval.Manual.load(ROOT / "index" / "royal-enfield-himalayan-450")
    row = next(i for i, c in enumerate(manual.chunks) if c["section"] == "TROUBLESHOOTING")
    # Pass the query vector in, so only the local index is touched.
    result = retrieval.search(manual, "engine does not start", query_vector=manual.vectors[row])
    assert "[p. 119]" in [r["cite_as"] for r in result["results"]]
