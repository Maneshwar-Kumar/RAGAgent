"""Download the demo's motorcycle manuals from their official sources and verify them.

    python scripts/fetch_manuals.py                 # download missing manuals, verify SHA-256
    python scripts/fetch_manuals.py --ingest        # ...then build index/<manual_id>/ for each
    python scripts/fetch_manuals.py --verify-only   # check manuals/ against sources.json (no network)

The two bundled manuals and their prebuilt index are committed (attribution in
manuals/README.md), so nothing needs to be downloaded or embedded after cloning. This script
checks them, restores a missing PDF from its official URL, and rebuilds an index only when
asked. manuals/sources.json records each official download URL and the SHA-256 of the exact
edition the index and the evals were built from. A file that does not match is never used:
the script reports it and exits with an error.

Downloading and verifying use only the Python standard library. --ingest calls the embedding
model, so it needs the project dependencies and the same credentials as the agent (env.sample).
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANUALS_DIR = ROOT / "manuals"
SOURCES = MANUALS_DIR / "sources.json"
# Same default as bike_manual_agent.config.INDEX_DIR. Not imported from there so that
# downloading and verifying work with nothing but the standard library.
INDEX_DIR = Path(os.getenv("RAGAGENT_INDEX_DIR", ROOT / "index"))

# Some OEM sites reject requests that do not look like they come from a browser.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
BLOCK = 1 << 20  # 1 MiB
REQUIRED_KEYS = ("manual_id", "name", "file", "url", "sha256")


class FetchError(Exception):
    """A manual could not be downloaded, or the download did not match its SHA-256."""


def say(msg: str) -> None:
    print(msg, flush=True)


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def load_sources(path: Path = SOURCES) -> list[dict]:
    """Read and sanity-check sources.json (file names and ids become local paths)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("manuals") or []
    if not entries:
        raise SystemExit(f"ERROR: {rel(path)} lists no manuals.")
    for e in entries:
        where = f"{rel(path)} entry {e.get('manual_id', '?')!r}"
        missing = [k for k in REQUIRED_KEYS if not e.get(k)]
        if missing:
            raise SystemExit(f"ERROR: {where} is missing: {', '.join(missing)}")
        if Path(e["file"]).name != e["file"] or not e["file"].lower().endswith(".pdf"):
            raise SystemExit(f"ERROR: {where}: 'file' must be a plain .pdf file name.")
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", e["manual_id"]):
            raise SystemExit(f"ERROR: {where}: 'manual_id' must be lowercase letters, digits, '-'.")
        if not e["url"].startswith("https://"):
            raise SystemExit(f"ERROR: {where}: 'url' must be an https:// URL.")
        if not re.fullmatch(r"[0-9a-f]{64}", e["sha256"].lower()):
            raise SystemExit(f"ERROR: {where}: 'sha256' must be 64 hex characters.")
        e["sha256"] = e["sha256"].lower()
    return entries


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, dest: Path, expected_sha: str, attempts: int = 3) -> int:
    """Download `url` to `dest` and return its size; raise FetchError on any failure.

    Bytes go to a temporary file next to `dest` (named so that .gitignore's manuals/*.pdf
    covers it) and are moved into place only once the SHA-256 matches, so an interrupted or
    altered download never leaves a plausible-looking manual behind.
    """
    tmp = dest.with_name(f".{dest.stem}.partial.pdf")
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/pdf,*/*;q=0.8"}
    )
    try:
        for attempt in range(1, attempts + 1):
            digest, size, content_type, is_pdf = hashlib.sha256(), 0, "", False
            try:
                with urllib.request.urlopen(request, timeout=60) as resp, tmp.open("wb") as out:
                    content_type = resp.headers.get("Content-Type", "")
                    while block := resp.read(BLOCK):
                        if size == 0:
                            is_pdf = block.startswith(b"%PDF")
                        digest.update(block)
                        out.write(block)
                        size += len(block)
                break
            except urllib.error.HTTPError as exc:
                if exc.code < 500 or attempt == attempts:  # 4xx will not fix itself
                    raise FetchError(f"HTTP {exc.code} {exc.reason}") from exc
            except (urllib.error.URLError, http.client.HTTPException, OSError) as exc:
                if attempt == attempts:
                    raise FetchError(f"download failed: {exc}") from exc
            time.sleep(2**attempt)

        actual = digest.hexdigest()
        if actual != expected_sha:
            hint = (
                f" The server did not return a PDF (Content-Type: {content_type or 'unknown'});"
                " it may be blocking scripted downloads."
                if not is_pdf
                else " The manufacturer may have replaced the file with a new edition."
            )
            raise FetchError(
                f"SHA-256 mismatch: expected {expected_sha}, got {actual} ({size:,} bytes).{hint}"
            )
        os.replace(tmp, dest)
        return size
    finally:
        tmp.unlink(missing_ok=True)


def index_status(entry: dict) -> str:
    """Describe index/<manual_id>/ relative to the pinned PDF (no model calls)."""
    meta_path = INDEX_DIR / entry["manual_id"] / "meta.json"
    if not meta_path.exists():
        return "not indexed yet (run with --ingest)"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "WARNING: meta.json is unreadable (re-run with --ingest)"
    if meta.get("source_sha256") != entry["sha256"]:
        return "WARNING: built from a different PDF (re-run with --ingest)"
    return f"indexed ({meta.get('chunks', '?')} chunks, {meta.get('embedding_model', '?')})"


def ingest_all(entries: list[dict]) -> list[str]:
    """Index each verified manual; skip ones whose index already matches. Returns failures."""
    sys.path.insert(0, str(ROOT))
    try:
        from bike_manual_agent import config
        from bike_manual_agent.ingest import ingest
    except ImportError as exc:
        say(f"ERROR: --ingest needs the project dependencies (pip install -r requirements.txt): {exc}")
        return [e["manual_id"] for e in entries]

    failed = []
    for e in entries:
        mid = e["manual_id"]
        folder = config.INDEX_DIR / mid
        try:
            meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
        current = (
            meta.get("source_sha256") == e["sha256"]
            and meta.get("name") == e["name"]
            and meta.get("embedding_model") == config.EMBEDDING_MODEL
            and meta.get("chunk_chars") == config.CHUNK_CHARS
            and meta.get("chunk_overlap") == config.CHUNK_OVERLAP
            and (folder / "chunks.json").exists()
            and (folder / "embeddings.npy").exists()
        )
        if current:
            say(f"[indexed]  {mid}: {rel(folder)}/ is up to date (delete it to rebuild)")
            continue
        say(f"[ingest]   {mid}: extracting text and embedding with {config.EMBEDDING_MODEL} ...")
        started = time.time()
        try:
            out = ingest(MANUALS_DIR / e["file"], e["name"], mid)
        except Exception as exc:  # credentials, quota, network, unreadable PDF
            say(f"[FAILED]   {mid}: indexing failed: {type(exc).__name__}: {exc}")
            say("           Check the model credentials described in env.sample.")
            failed.append(mid)
            continue
        meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
        say(
            f"[ok]       {mid}: {meta['pdf_pages']} pages -> {meta['chunks']} chunks "
            f"in {rel(out)}/ ({time.time() - started:.0f} s)"
        )
    return failed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--verify-only",
        action="store_true",
        help="only check the PDFs already in manuals/ against sources.json; download nothing",
    )
    mode.add_argument(
        "--ingest",
        action="store_true",
        help="after downloading and verifying, index each manual (calls the embedding model; "
        "manuals whose index is already up to date are skipped)",
    )
    args = parser.parse_args(argv)

    entries = load_sources()
    MANUALS_DIR.mkdir(exist_ok=True)
    verified: list[dict] = []
    failed: list[str] = []

    for e in entries:
        mid, path, expected = e["manual_id"], MANUALS_DIR / e["file"], e["sha256"]
        if path.exists():
            actual = sha256_of(path)
            if actual == expected:
                say(f"[ok]       {mid}: {rel(path)} matches SHA-256 {expected[:16]}...")
                verified.append(e)
            else:
                say(f"[MISMATCH] {mid}: {rel(path)} has SHA-256 {actual}")
                say(f"           expected {expected} ({e.get('edition', 'pinned edition')}).")
                say("           This is not the edition the index and evals were built from. Move")
                say("           or delete the file and re-run to download the pinned edition, or, to")
                say("           adopt a new edition deliberately, update sources.json and re-ingest.")
                failed.append(mid)
            continue
        if args.verify_only:
            say(f"[missing]  {mid}: {rel(path)} not found (run without --verify-only to download)")
            failed.append(mid)
            continue
        say(f"[download] {mid}: {e['url']}")
        started = time.time()
        try:
            size = download(e["url"], path, expected)
        except FetchError as exc:
            say(f"[FAILED]   {mid}: {exc}")
            say(f"           You can download it in a browser from the URL above, save it as")
            say(f"           {rel(path)}, then run: python scripts/fetch_manuals.py --verify-only")
            failed.append(mid)
            continue
        say(
            f"[ok]       {mid}: saved {rel(path)} ({size / 1e6:.1f} MB in "
            f"{time.time() - started:.0f} s), SHA-256 verified"
        )
        verified.append(e)

    if args.ingest and verified:
        failed += ingest_all(verified)
    elif verified:
        for e in verified:
            say(f"           {e['manual_id']} index: {index_status(e)}")

    if failed:
        print(f"ERROR: {len(failed)} manual(s) failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    say(f"All {len(entries)} manual(s) verified" + (" and indexed." if args.ingest else "."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
