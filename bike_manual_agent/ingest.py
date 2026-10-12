"""Turn a bike owner's/service manual PDF into a searchable, page-cited index.

Usage:
    python ingest.py manuals/my_manual.pdf --name "TVS Apache RTR 160 4V"

The index is written to index/<manual_id>/ and contains:
    meta.json        manual name, source file hash, page count, models used
    chunks.json      text chunks with PDF page, printed page and section heading
    embeddings.npy   one normalised embedding per chunk

Ingestion runs once per manual; questions only embed the query.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import config

_HEADING_RE = re.compile(r"^[A-Z0-9][A-Z0-9 &/,'()\-–.:]{3,}$")


@dataclass
class Chunk:
    chunk_id: str
    pdf_page: int  # 1-based page index in the PDF file
    printed_page: str | None  # page number printed on the page, if detected
    section: str | None  # nearest upper-case heading on the page
    text: str


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:60] or "manual"


def extract_pages(pdf_path: Path) -> list[str]:
    import pymupdf

    with pymupdf.open(pdf_path) as doc:
        return [page.get_text() for page in doc]


def _page_labels(text: str) -> tuple[str | None, str | None]:
    """Detect the printed page number and section heading near the top of a page."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    printed = next((ln for ln in lines[:4] if re.fullmatch(r"\d{1,4}", ln)), None)
    section = next((ln for ln in lines[:8] if _HEADING_RE.match(ln) and not ln.isdigit()), None)
    return printed, section


def _clean(text: str) -> str:
    text = re.sub(r"[-]", "-", text)  # private-use bullet glyphs
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_pages(pages: list[str], size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page_no, raw in enumerate(pages, start=1):
        text = _clean(raw)
        if len(text) < 40:  # blank or image-only page
            continue
        printed, section = _page_labels(raw)
        start, part = 0, 0
        while start < len(text):
            end = min(len(text), start + size)
            if end < len(text):  # prefer to break on a line boundary
                cut = text.rfind("\n", start + size // 2, end)
                end = cut if cut > 0 else end
            piece = text[start:end].strip()
            if piece:
                chunks.append(Chunk(f"p{page_no}-{part}", page_no, printed, section, piece))
                part += 1
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
    return chunks


def embed_texts(texts: list[str], task_type: str, batch_size: int = 32) -> np.ndarray:
    from google.genai import types

    client = config.genai_client()
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        for attempt in range(5):
            try:
                resp = client.models.embed_content(
                    model=config.EMBEDDING_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(task_type=task_type),
                )
                break
            except Exception:  # transient quota / network errors
                if attempt == 4:
                    raise
                time.sleep(2**attempt)
        vectors.extend(e.values for e in resp.embeddings)
    arr = np.asarray(vectors, dtype=np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True).clip(min=1e-9)


def ingest(pdf_path: Path, name: str | None = None, manual_id: str | None = None) -> Path:
    pdf_path = Path(pdf_path)
    pages = extract_pages(pdf_path)
    chunks = chunk_pages(pages, config.CHUNK_CHARS, config.CHUNK_OVERLAP)
    if not chunks:
        raise ValueError(
            f"No extractable text in {pdf_path.name}. Scanned (image-only) PDFs need OCR first."
        )
    name = name or pdf_path.stem.replace("-", " ").replace("_", " ").title()
    manual_id = manual_id or slugify(name)
    # Section and page context is embedded with the text so headings help recall.
    embed_inputs = [f"{c.section or ''}\n{c.text}" for c in chunks]
    vectors = embed_texts(embed_inputs, "RETRIEVAL_DOCUMENT")

    out = config.INDEX_DIR / manual_id
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "manual_id": manual_id,
        "name": name,
        "source_file": pdf_path.name,
        "source_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
        "pdf_pages": len(pages),
        "chunks": len(chunks),
        "embedding_model": config.EMBEDDING_MODEL,
        "chunk_chars": config.CHUNK_CHARS,
        "chunk_overlap": config.CHUNK_OVERLAP,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out / "chunks.json").write_text(
        json.dumps([asdict(c) for c in chunks], ensure_ascii=False, indent=1), encoding="utf-8"
    )
    np.save(out / "embeddings.npy", vectors)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdf", type=Path, help="Path to the manual PDF")
    parser.add_argument("--name", help='Display name, e.g. "Royal Enfield Himalayan 450"')
    parser.add_argument("--id", dest="manual_id", help="Index folder name (default: slug of --name)")
    args = parser.parse_args()
    out = ingest(args.pdf, args.name, args.manual_id)
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    print(f"Indexed '{meta['name']}': {meta['pdf_pages']} pages -> {meta['chunks']} chunks at {out}")


if __name__ == "__main__":
    main()
