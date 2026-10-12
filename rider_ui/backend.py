"""Wiring between the MotoMedic rider UI and the ADK agent, which runs in-process.

Streamlit re-runs its script on worker threads, so the ADK Runner lives on one long-lived
asyncio loop in a background thread. The UI submits a question, receives a Future, and
reads live progress (the agent's actual tool calls) from a queue while it waits.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import queue
import re
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from bike_manual_agent import agent, config, retrieval
from bike_manual_agent.ingest import ingest

log = logging.getLogger("rider_ui")

APP_NAME = "motomedic_rider_ui"
SESSION_IDLE_SECONDS = 2 * 60 * 60
SOURCES_FILE = config.REPO_ROOT / "manuals" / "sources.json"
ERROR_TEXT = (
    "MotoMedic couldn't finish checking the manual just now, so no answer was given. "
    "Please try again in a moment."
)
EXPIRED_TEXT = (
    "This conversation expired after a period of inactivity. Start a new conversation to "
    "continue."
)


@dataclass(frozen=True)
class Manual:
    manual_id: str
    name: str
    model_label: str
    short_name: str
    edition: str
    edition_short: str
    brand: str | None
    official_site: str | None
    source_url: str | None
    select_label: str
    pill_label: str
    confirm_label: str
    source_title: str
    edition_label: str
    pages: int
    chunks: int
    uploaded: bool = False
    upload_sha: str | None = None


@dataclass(frozen=True)
class Citation:
    label: str  # as cited, e.g. "p. 119"
    pdf_page: int
    section: str | None
    excerpt: str


@dataclass(frozen=True)
class Answer:
    status: str  # answered | not_covered | need_info | guardrail | error
    body: str  # markdown without the status line
    citations: tuple[Citation, ...]
    searches: tuple[str, ...]
    elapsed_s: float
    reference: str


def _sources() -> dict[str, dict]:
    try:
        data = json.loads(SOURCES_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {m["manual_id"]: m for m in data.get("manuals", [])}


def describe_manual(manual_id: str, upload_sha: str | None = None) -> Manual:
    meta = retrieval.load_manual(manual_id).meta
    src = _sources().get(manual_id, {})
    pages = int(meta.get("pdf_pages", 0))
    uploaded = manual_id.startswith(agent.UPLOAD_PREFIX)
    return Manual(
        manual_id=manual_id,
        name=src.get("name", meta["name"]),
        model_label=src.get("model_label", meta["name"]),
        short_name=src.get("short_name", "motorcycle"),
        edition=src.get("edition", f"{pages} pages"),
        edition_short=src.get("edition_short", "Your uploaded manual" if uploaded else "Owner’s manual"),
        brand=src.get("brand"),
        official_site=src.get("official_site"),
        source_url=src.get("url"),
        select_label=src.get("select_label", meta["name"]),
        pill_label=src.get("pill_label", meta["name"]),
        confirm_label=src.get("confirm_label", "This manual is for my motorcycle"),
        source_title=src.get("source_title", meta["name"]),
        edition_label=src.get(
            "edition_label", f"Uploaded manual · {pages} pages" if uploaded else f"{pages} pages"
        ),
        pages=pages,
        chunks=int(meta.get("chunks", 0)),
        uploaded=uploaded,
        upload_sha=upload_sha,
    )


def builtin_manuals() -> list[Manual]:
    return [
        describe_manual(mid)
        for mid in retrieval.list_manual_ids()
        if not mid.startswith(agent.UPLOAD_PREFIX)
    ]


def parse_answer(text: str) -> tuple[str, str]:
    """Split the agent's reply into (status, body without the status line)."""
    stripped = agent.status_first(text.strip())
    first, _, rest = stripped.partition("\n")
    status = agent.reply_status(first)
    body = rest.strip() if status else stripped
    if "(guardrail:" in stripped:
        return "guardrail", body
    if status:
        return status, body
    return ("answered" if agent._cited_labels(stripped) else "need_info"), body


def _excerpt(result: dict, limit: int = 320) -> str:
    """Passage text without the page's running header (brand line, page number, heading)."""
    lines = [ln.strip() for ln in result["text"].splitlines() if ln.strip()]
    printed, section = result.get("printed_page"), result.get("section")
    if printed and printed in lines[:3]:
        lines = lines[lines.index(printed) + 1 :]
    if lines and section and lines[0] == section:
        lines = lines[1:]
    flat = " ".join(" ".join(lines).split())
    if flat[:1].islower():  # overlapping chunks can start mid-sentence: skip to a clean start
        cut = min((i for i in (flat.find(". "), flat.find(" - ")) if 0 <= i < 160), default=-1)
        flat = "…" + flat[cut + 1 :].lstrip(" -") if cut >= 0 else "…" + flat.split(" ", 1)[-1]
    return flat if len(flat) <= limit else flat[: limit - 1].rsplit(" ", 1)[0] + "…"


class Backend:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, name="adk-loop", daemon=True).start()
        self.sessions = InMemorySessionService()
        self.runner = Runner(agent=agent.root_agent, app_name=APP_NAME, session_service=self.sessions)
        self._lock = threading.Lock()
        self._upload_lock = threading.Lock()
        self._last_used: dict[tuple[str, str], float] = {}
        self._retrieved: dict[str, dict[str, dict]] = {}  # session -> label -> search result

    # ------------------------------------------------------------ manuals

    def manuals(self) -> list[Manual]:
        return builtin_manuals()

    def add_manual(self, data: bytes, filename: str, limits) -> Manual:
        """Index a rider-supplied PDF manual. Raises ValueError with a rider-facing reason."""
        import pymupdf

        if not data.startswith(b"%PDF"):
            raise ValueError("That file doesn't look like a PDF.")
        if len(data) > limits.manual_megabytes * 1024 * 1024:
            raise ValueError(f"Manuals can be up to {limits.manual_megabytes} MB.")
        try:
            with pymupdf.open(stream=data, filetype="pdf") as doc:
                pages = doc.page_count
        except Exception:
            raise ValueError("That PDF couldn't be opened.") from None
        if pages > limits.manual_pages:
            raise ValueError(f"Manuals can have up to {limits.manual_pages} pages ({pages} here).")
        sha = hashlib.sha256(data).hexdigest()
        manual_id = f"{agent.UPLOAD_PREFIX}{sha[:10]}"
        with self._upload_lock:
            if manual_id not in retrieval.list_manual_ids():
                config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
                path = config.UPLOAD_DIR / f"{manual_id}.pdf"
                path.write_bytes(data)
                name = re.sub(r"[_\-]+", " ", Path(filename).stem).strip() or "Uploaded manual"
                ingest(path, name[:80], manual_id)  # ValueError for scanned PDFs
        return describe_manual(manual_id, upload_sha=sha)

    # ------------------------------------------------------------ conversations

    def _call(self, coro, timeout: float = 30):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def start_conversation(self, user_id: str, manual: Manual) -> str:
        state = {agent.SELECTED: manual.manual_id}
        if manual.uploaded and manual.upload_sha:
            state[agent.UPLOADS] = {manual.upload_sha: manual.manual_id}
        session = self._call(
            self.sessions.create_session(app_name=APP_NAME, user_id=user_id, state=state)
        )
        self._touch(user_id, session.id)
        return session.id

    def _touch(self, user_id: str, session_id: str) -> None:
        now = time.monotonic()
        with self._lock:
            self._last_used[(user_id, session_id)] = now
            stale = [k for k, t in self._last_used.items() if now - t > SESSION_IDLE_SECONDS]
            for key in stale:
                del self._last_used[key]
                self._retrieved.pop(key[1], None)
        for user, sid in stale:
            asyncio.run_coroutine_threadsafe(
                self.sessions.delete_session(app_name=APP_NAME, user_id=user, session_id=sid),
                self.loop,
            )

    def ask(
        self,
        user_id: str,
        session_id: str,
        text: str,
        photo: tuple[bytes, str] | None,
        progress: "queue.Queue[str]",
    ) -> Future:
        self._touch(user_id, session_id)
        return asyncio.run_coroutine_threadsafe(
            self._ask(user_id, session_id, text, photo, progress), self.loop
        )

    async def _ask(self, user_id, session_id, text, photo, progress) -> Answer:
        started = time.monotonic()
        parts = []
        if photo:
            data, mime = photo
            parts.append(types.Part(inline_data=types.Blob(mime_type=mime, data=data)))
            progress.put("Looking at your photo…")
        parts.append(types.Part(text=text))
        if await self.sessions.get_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id
        ) is None:
            return Answer("error", EXPIRED_TEXT, (), (), 0.0, "")
        retrieved = self._retrieved.setdefault(session_id, {})
        searches: list[str] = []
        final_text, reference = "", ""
        try:
            async for event in self.runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=types.Content(role="user", parts=parts),
            ):
                reference = reference or event.invocation_id or ""
                for call in event.get_function_calls():
                    if call.name == "search_manual":
                        query = str((call.args or {}).get("query", ""))
                        searches.append(query)
                        progress.put(f"Searching the manual: “{query}”")
                    elif call.name == "select_manual":
                        progress.put("Selecting the manual…")
                for resp in event.get_function_responses():
                    if resp.name == "search_manual":
                        for result in (resp.response or {}).get("results", []):
                            label = agent._normalise_labels(result["cite_as"])
                            retrieved.setdefault(label, result)
                        progress.put("Reading the manual excerpts…")
                if event.is_final_response() and event.content and event.content.parts:
                    final_text = "".join(
                        p.text or "" for p in event.content.parts if not getattr(p, "thought", False)
                    )
        except Exception:  # provider outage, quota, malformed output...
            log.exception("agent run failed")
            final_text = ""
        elapsed = round(time.monotonic() - started, 1)
        if not final_text.strip():
            return Answer("error", ERROR_TEXT, (), tuple(searches), elapsed, reference)

        status, body = parse_answer(final_text)
        citations, seen = [], set()
        for label in agent._cited_labels(body):
            result = retrieved.get(label)
            if result is None or label in seen:
                continue
            seen.add(label)
            citations.append(
                Citation(
                    label=result["cite_as"].strip("[]"),
                    pdf_page=int(result["pdf_page"]),
                    section=result.get("section"),
                    excerpt=_excerpt(result),
                )
            )
        log.info(
            "answer status=%s elapsed=%.1fs searches=%d citations=%d",
            status, elapsed, len(searches), len(citations),
        )
        return Answer(status, body, tuple(citations), tuple(searches), elapsed, reference)


_backend: Backend | None = None
_backend_lock = threading.Lock()


def get_backend() -> Backend:
    global _backend
    with _backend_lock:
        if _backend is None:
            _backend = Backend()
        return _backend
