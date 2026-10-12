"""Evaluate the bike-manual troubleshooter against evals/cases.json.

    python evals/run_eval.py                                  # every case, LLM judge on, 4 at a time
    python evals/run_eval.py --only re-misfire tvs-abs-lamp   # targeted re-run (written to a separate file)
    python evals/run_eval.py --only re-misfire --merge        # ...and merge it into latest.json + the report
    python evals/run_eval.py --no-judge --concurrency 2

Needs the same environment as the agent (Vertex AI: GOOGLE_GENAI_USE_VERTEXAI, GOOGLE_CLOUD_PROJECT,
GOOGLE_CLOUD_LOCATION and ADC, or a short-lived GOOGLE_OAUTH_ACCESS_TOKEN).

Every case runs in a fresh session with its own Runner. A Runner plugin records each raw model
response *before* the agent's citation guardrail can replace it, which gives token usage and the
pre-guardrail text. A different, stronger model then judges faithfulness of the final answer
against exactly the excerpts the agent retrieved.

Outputs
    evals/results/latest.json         full results without manual text or image bytes
    evals/results/latest.raw.json     same plus the retrieved excerpt texts (gitignored)
    evals/results/<run_id>*.raw.json  per-run copy (gitignored)
    docs/EVAL_RESULTS.md              report (a hand-written analysis block between the
                                      analysis markers is preserved when it is regenerated)
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import math
import mimetypes
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import bike_manual_agent  # noqa: E402,F401  (installs token credentials when GOOGLE_OAUTH_ACCESS_TOKEN is set)
from bike_manual_agent import config, retrieval  # noqa: E402
from bike_manual_agent.agent import (  # noqa: E402
    SELECTED,
    _cited_labels,
    _normalise_labels,
    root_agent,
)
from google.adk.plugins.base_plugin import BasePlugin  # noqa: E402
from google.adk.runners import Runner  # noqa: E402
from google.adk.sessions import InMemorySessionService  # noqa: E402
from google.genai import types  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

CASES_FILE = ROOT / "evals" / "cases.json"
RESULTS_DIR = ROOT / "evals" / "results"
REPORT_FILE = ROOT / "docs" / "EVAL_RESULTS.md"
APP_NAME = "bike_manual_agent"
DEFAULT_JUDGE = "gemini-3.1-pro-preview"
FALLBACK_JUDGE = "gemini-2.5-pro"
STATUSES = ("answered", "not_covered", "need_info")
STATUS_WORDS = {
    "from the manual": "answered",
    "not covered by the manual": "not_covered",
    "need more information": "need_info",
}
EXACT_STATUS_LINES = {"**From the manual**", "**Not covered by the manual**", "**Need more information**"}
STATUS_RE = re.compile(
    r"^[\s>#*_]*(from the manual|not covered by the manual|need more information)\b", re.IGNORECASE
)
ANALYSIS_START = "<!-- analysis:start -->"
ANALYSIS_END = "<!-- analysis:end -->"
SHORT_MANUAL = {"royal-enfield-himalayan-450": "RE Himalayan 450", "tvs-ronin": "TVS Ronin"}

ABORT: dict = {"reason": None}  # set on authentication failure so queued cases are skipped


def log(msg: str) -> None:
    print(msg, flush=True)


def agent_model_name() -> str:
    model = root_agent.model  # a model name, or a Gemini(...) object carrying retry options
    return model.model if hasattr(model, "model") else str(model)


def _clean_model_name(value) -> str:
    """Older result files stored the Gemini object's repr instead of its model name."""
    match = re.search(r"model='([^']+)'", str(value))
    return match.group(1) if match else str(value)


_TOKEN = {"refreshed_at": 0.0, "count": 0}
_TOKEN_LOCK = __import__("threading").Lock()


def refresh_access_token() -> bool:
    """On a 401 in token mode, fetch a new token from gcloud and reinstall the credentials.

    `gcloud auth print-access-token` can hand out a cached token with only minutes left, so a
    long run may outlive it. The token only lives in this process's memory - it is never
    printed, logged or written. Every case builds new clients, so later calls pick it up.
    """
    if not os.getenv("GOOGLE_OAUTH_ACCESS_TOKEN"):
        return False  # ADC mode refreshes by itself
    import subprocess

    with _TOKEN_LOCK:
        if time.time() - _TOKEN["refreshed_at"] < 60:
            return True  # another case just refreshed it
        token = ""
        for cmd in ('gcloud config config-helper --min-expiry=55m --format="value(credential.access_token)"',
                    "gcloud auth print-access-token"):
            try:
                out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
            except Exception:  # noqa: BLE001
                continue
            token = out.stdout.strip() if out.returncode == 0 else ""
            if token and " " not in token:
                break
        if not token or " " in token:
            return False
        os.environ["GOOGLE_OAUTH_ACCESS_TOKEN"] = token
        config.use_access_token_if_provided()
        # Drop every cached client built with the old token.
        for cached in (getattr(config, "genai_client", None), getattr(retrieval, "embed_query", None)):
            if hasattr(cached, "cache_clear"):
                cached.cache_clear()
        model_obj = root_agent.model
        if hasattr(model_obj, "__dict__"):
            for key in ("_per_loop_api_client", "api_client"):
                model_obj.__dict__.pop(key, None)
        _TOKEN.update(refreshed_at=time.time(), count=_TOKEN["count"] + 1)
        log("    access token refreshed after a 401")
        return True


# ------------------------------------------------------------------ hygiene / errors


def scrub(text) -> str:
    """Remove the project id and anything that looks like a credential from recorded text."""
    text = str(text)
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    if project:
        text = text.replace(project, "<project>")
    text = re.sub(r"ya29\.[A-Za-z0-9_\-.]+", "<token>", text)
    text = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9_\-.]+", r"\1<token>", text)
    text = re.sub(r"AIza[0-9A-Za-z_\-]{20,}", "<api-key>", text)
    return text


def _exc_chain(exc):
    seen, stack = set(), [exc]
    while stack:
        e = stack.pop()
        if e is None or id(e) in seen:
            continue
        seen.add(id(e))
        yield e
        stack.extend(getattr(e, "exceptions", None) or ())
        stack.extend([e.__cause__, e.__context__])


_TRANSIENT_CODES = {408, 429, 500, 502, 503, 504}
_TRANSIENT_NAMES = (
    "timeout", "connecterror", "connectionerror", "remoteprotocol", "readerror", "writeerror",
    "networkerror", "transporterror", "serverdisconnected", "clientconnector", "gaierror",
    "connectionreset", "brokenpipe",
)
_TRANSIENT_TEXT = (
    "resource_exhausted", "resource exhausted", "unavailable", "deadline", "timed out", "timeout",
    "temporarily", "connection reset", "econnreset", "connection aborted", "server disconnected",
    "getaddrinfo", "name resolution", "name or service not known", "nodename nor servname",
    "network is unreachable", "eof occurred", "too many requests", "try again", "overloaded",
)


def classify_error(exc) -> str:
    """'transient' (retry), 'auth' (stop the run) or 'fatal' (record and move on)."""
    from google.genai import errors as genai_errors

    for e in _exc_chain(exc):
        if isinstance(e, genai_errors.APIError):
            code = getattr(e, "code", None)
            if code in _TRANSIENT_CODES:
                return "transient"
            if code == 401:
                return "auth"
        if isinstance(e, (ConnectionError, TimeoutError, asyncio.TimeoutError)):
            return "transient"
        if any(k in type(e).__name__.lower() for k in _TRANSIENT_NAMES):
            return "transient"
    text = " ".join(f"{type(e).__name__}: {e}" for e in _exc_chain(exc)).lower()
    if "unauthenticated" in text or "invalid authentication credentials" in text:
        return "auth"
    if any(m in text for m in _TRANSIENT_TEXT) or re.search(r"\b(429|500|502|503|504)\b", text):
        return "transient"
    return "fatal"


def _model_unavailable(exc) -> bool:
    """The judge model does not exist / is not enabled here: fall back to the next one."""
    from google.genai import errors as genai_errors

    for e in _exc_chain(exc):
        if isinstance(e, genai_errors.APIError):
            msg = str(e).lower()
            if e.code in (403, 404) or (e.code == 400 and ("not found" in msg or "not supported" in msg)):
                return True
    return False


async def _backoff(attempt: int, what: str) -> None:
    delay = min(90.0, 5.0 * 2 ** (attempt - 1)) + random.uniform(0, 3)
    log(f"    {what}: transient error, retrying in {delay:.0f}s")
    await asyncio.sleep(delay)


# ------------------------------------------------------------------ cases


def load_cases(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data["cases"] if isinstance(data, dict) else data
    known = set(retrieval.list_manual_ids())
    ids = set()
    for c in cases:
        cid = c.get("id")
        problems = []
        if not cid or cid in ids:
            problems.append("missing or duplicate id")
        ids.add(cid)
        if c.get("expected_status") not in STATUSES:
            problems.append(f"expected_status must be one of {STATUSES}")
        if c.get("manual_id") is not None and c["manual_id"] not in known:
            problems.append(f"unknown manual_id {c['manual_id']!r} (indexed: {sorted(known)})")
        if c.get("attachment") and not (ROOT / c["attachment"]).exists():
            problems.append(f"attachment not found: {c['attachment']}")
        if not c.get("question"):
            problems.append("missing question")
        for key in ("expected_pages", "must_mention", "must_not_mention"):
            c.setdefault(key, [])
        if problems:
            raise SystemExit(f"Bad case {cid!r}: {'; '.join(problems)}")
    return cases


# ------------------------------------------------------------------ running the agent


class CapturePlugin(BasePlugin):
    """Records every raw model response before the agent's callbacks (and guardrail) run,
    plus how long each model call and each tool call took."""

    def __init__(self):
        super().__init__(name="eval_capture")
        self.calls: list[dict] = []
        self.tool_times: list[dict] = []
        self._model_started: float | None = None
        self._tool_started: dict[int, float] = {}

    async def before_model_callback(self, *, callback_context, llm_request):
        self._model_started = time.perf_counter()
        return None

    async def before_tool_callback(self, *, tool, tool_args, tool_context):
        self._tool_started[id(tool_context)] = time.perf_counter()
        return None

    async def after_tool_callback(self, *, tool, tool_args, tool_context, result):
        started = self._tool_started.pop(id(tool_context), None)
        if started is not None:
            self.tool_times.append({"tool": tool.name, "seconds": round(time.perf_counter() - started, 2)})
        return None

    async def after_model_callback(self, *, callback_context, llm_response):
        if llm_response.partial:
            return None
        seconds = round(time.perf_counter() - self._model_started, 2) if self._model_started else None
        parts = llm_response.content.parts if llm_response.content and llm_response.content.parts else []
        usage = {}
        u = llm_response.usage_metadata
        for key in (
            "prompt_token_count", "candidates_token_count", "thoughts_token_count",
            "cached_content_token_count", "tool_use_prompt_token_count", "total_token_count",
        ):
            value = getattr(u, key, None) if u is not None else None
            if value:
                usage[key] = int(value)
        self.calls.append(
            {
                "model_version": llm_response.model_version,
                "text": "".join(p.text for p in parts if p.text and not p.thought),
                "function_calls": [p.function_call.name for p in parts if p.function_call],
                "finish_reason": str(llm_response.finish_reason) if llm_response.finish_reason else None,
                "error_code": llm_response.error_code,
                "usage": usage,
                "seconds": seconds,
            }
        )
        return None  # never alter the agent's behaviour


class TransientEventError(RuntimeError):
    pass


async def run_agent_once(case: dict) -> dict:
    plugin = CapturePlugin()
    runner = Runner(
        agent=root_agent, app_name=APP_NAME, session_service=InMemorySessionService(), plugins=[plugin]
    )
    try:
        state = {SELECTED: case["manual_id"]} if case.get("manual_id") else {}
        session = await runner.session_service.create_session(app_name=APP_NAME, user_id="eval", state=state)
        parts = [types.Part(text=case["question"])]
        if case.get("attachment"):
            path = ROOT / case["attachment"]
            mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
            blob = types.Blob(mime_type=mime, data=path.read_bytes(), display_name=path.name)
            parts.insert(0, types.Part(inline_data=blob))
        tool_calls, searches, other_responses, event_errors, answer = [], [], [], [], ""
        started = time.perf_counter()
        async for event in runner.run_async(
            user_id="eval", session_id=session.id, new_message=types.Content(role="user", parts=parts)
        ):
            if event.error_code:
                code = str(event.error_code)
                event_errors.append({"error_code": code, "error_message": scrub(event.error_message or "")[:300]})
                if any(k in code.upper() for k in ("429", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "503", "500")):
                    raise TransientEventError(f"{code}: {event.error_message}")
            for call in event.get_function_calls():
                tool_calls.append({"name": call.name, "args": dict(call.args or {})})
            for resp in event.get_function_responses():
                if resp.name == "search_manual":
                    searches.append(resp.response or {})
                else:
                    other_responses.append({"name": resp.name, "response": resp.response})
            if event.is_final_response() and event.content and event.content.parts:
                answer = "".join(p.text for p in event.content.parts if p.text and not p.thought)
        latency = time.perf_counter() - started
        final = await runner.session_service.get_session(app_name=APP_NAME, user_id="eval", session_id=session.id)
        final_state = dict(final.state) if final else {}
    finally:
        close = getattr(runner, "close", None)
        if close:
            try:
                await close()
            except Exception:  # noqa: BLE001 - closing must never fail a case
                pass
        try:  # close this loop's model client before the loop ends (avoids 'Event loop is closed' noise)
            await root_agent.canonical_model.api_client.aio.aclose()
        except Exception:  # noqa: BLE001
            pass
    return {
        "answer": answer.strip(),
        "latency_s": round(latency, 2),
        "tool_calls": tool_calls,
        "searches": searches,
        "other_tool_responses": other_responses,
        "event_errors": event_errors,
        "model_calls": plugin.calls,
        "tool_times": plugin.tool_times,
        "model_seconds": round(sum(c["seconds"] or 0 for c in plugin.calls), 2),
        "tool_seconds": round(sum(t["seconds"] for t in plugin.tool_times), 2),
        "final_selected_manual": final_state.get(SELECTED),
    }


def _run_case_on_own_loop(case: dict, timeout: float) -> dict:
    """search_manual is a synchronous tool that makes a blocking embedding call, and ADK runs
    sync tools on the event loop. Giving every case its own loop (in a worker thread) keeps one
    case's blocking call from stalling the others, so per-question latency is not inflated by
    the harness's own concurrency. ADK caches its model client per event loop, so this is safe."""
    return asyncio.run(asyncio.wait_for(run_agent_once(case), timeout=timeout))


async def run_agent_with_retry(case: dict, retries: int, timeout: float) -> dict:
    errors = []
    for attempt in range(1, retries + 2):
        if ABORT["reason"]:
            return {"error": f"skipped: {ABORT['reason']}", "error_kind": "skipped", "attempts": attempt - 1,
                    "transient_errors": errors}
        try:
            res = await asyncio.to_thread(_run_case_on_own_loop, case, timeout)
            res.update(attempts=attempt, transient_errors=errors)
            return res
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, (TransientEventError, asyncio.TimeoutError, TimeoutError)):
                kind = "transient"
                if not str(exc):
                    exc = TimeoutError(f"case exceeded {timeout:.0f}s")
            else:
                kind = classify_error(exc)
            msg = scrub(f"{type(exc).__name__}: {exc}")[:600]
            errors.append(f"attempt {attempt}: {kind}: {msg}")
            if kind == "auth":
                if attempt <= retries and await asyncio.to_thread(refresh_access_token):
                    continue  # retry this case with the new token
                ABORT["reason"] = "authentication failed (expired token?) - re-export GOOGLE_OAUTH_ACCESS_TOKEN"
            if kind != "transient" or attempt > retries:
                return {"error": msg, "error_kind": kind, "attempts": attempt, "transient_errors": errors}
            await _backoff(attempt, case["id"])
    raise AssertionError("unreachable")


# ------------------------------------------------------------------ deterministic checks


def parse_status(answer: str) -> tuple[str, bool]:
    """(status, status line exactly as specified)."""
    if not answer.strip():
        return "empty", False
    if "(guardrail:" in answer:
        return "guardrail", True
    first = next(line for line in answer.splitlines() if line.strip()).strip()
    match = STATUS_RE.match(first)
    if not match:
        return "unparsed", False
    return STATUS_WORDS[match.group(1).lower()], first in EXACT_STATUS_LINES


def _norm(text: str) -> str:
    text = text.lower().replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    return re.sub(r"\s+", " ", text)


def check_must_mention(answer: str, items: list) -> list[dict]:
    text = _norm(answer)
    out = []
    for item in items:
        alternatives = [item] if isinstance(item, str) else list(item)
        found = next((a for a in alternatives if _norm(a) in text), None)
        out.append({"item": alternatives, "found": found is not None, "matched": found})
    return out


def _forbidden_pattern(term: str) -> re.Pattern:
    words = [re.escape(w) for w in re.split(r"[\s\-]+", term.strip().lower()) if w]
    pattern = r"(?<![a-z0-9])" + r"[\s\-]+".join(words)
    if term.strip()[-1:].isdigit():
        pattern += r"(?![0-9])"
    return re.compile(pattern)


def check_must_not_mention(answer: str, terms: list[str]) -> list[dict]:
    text = _norm(answer)
    hits = []
    for term in terms:
        match = _forbidden_pattern(term).search(text)
        if match:
            start, end = max(0, match.start() - 80), min(len(text), match.end() + 80)
            hits.append({"term": term, "context": text[start:end]})
    return hits


def _unique(seq):
    return list(dict.fromkeys(seq))


def _page_key(label: str) -> str:
    label = str(label).strip()
    m = re.fullmatch(r"(?i)pdf\s*(?:p\.?)?\s*(\S+)", label)
    return f"pdfp.{m.group(1).lower()}" if m else f"p.{label.lower()}"


def evaluate(case: dict, res: dict) -> None:
    """Fill in the deterministic checks of one case result (in place)."""
    answer = res.get("answer", "")
    excerpts, retrieved, search_log = {}, [], []
    for s in res.get("searches", []):
        results = s.get("results") or []
        search_log.append({"query": s.get("query"), "error": s.get("error"),
                           "labels": [r.get("cite_as") for r in results],
                           "chunk_ids": [r.get("chunk_id") for r in results]})
        for r in results:
            retrieved.append(_normalise_labels(r["cite_as"]))
            excerpts.setdefault(r["chunk_id"], {
                "chunk_id": r["chunk_id"], "cite_as": r["cite_as"], "location": r.get("location"),
                "section": r.get("section"), "similarity": r.get("similarity"), "text": r.get("text", ""),
            })
    res["search_queries"] = [s["query"] for s in search_log if s["query"]]
    res["search_log"] = search_log
    res["excerpts"] = list(excerpts.values())
    res["retrieved_labels"] = sorted(set(retrieved))

    raw_texts = [c["text"] for c in res.get("model_calls", []) if c["text"].strip() and not c["function_calls"]]
    raw = raw_texts[-1].strip() if raw_texts else ""
    # Kept only when the agent's after_model_callback changed the model's reply: either the
    # citation guardrail replaced it, or `status_first` moved a misplaced status line to the top.
    res["raw_model_answer"] = raw if raw and raw != answer else None
    res["reply_rewritten"] = None
    if res["raw_model_answer"]:
        res["reply_rewritten"] = "guardrail" if "(guardrail:" in answer else "status_line_moved"
        res["raw_status"] = parse_status(raw)[0]
        res["raw_cited_labels"] = _unique(_cited_labels(raw))
        res["raw_invalid_citations"] = [c for c in res["raw_cited_labels"] if c not in set(retrieved)]

    usage = {}
    for call in res.get("model_calls", []):
        for k, v in call["usage"].items():
            usage[k] = usage.get(k, 0) + v
    res["usage"] = usage
    res["n_model_calls"] = len(res.get("model_calls", []))
    res["model_versions"] = _unique(c["model_version"] for c in res.get("model_calls", []) if c["model_version"])
    score(case, res)


def score(case: dict, res: dict) -> None:
    """The case-dependent checks; also used by --rescore on stored answers."""
    answer = res.get("answer", "")
    if "error" in res:
        res["status"], res["status_line_exact"] = "error", False
    else:
        res["status"], res["status_line_exact"] = parse_status(answer)
    retrieved = set(res.get("retrieved_labels", []))
    res["cited_labels"] = _unique(_cited_labels(answer)) if answer else []
    res["invalid_citations"] = [c for c in res["cited_labels"] if c not in retrieved]
    expected_pages = {_page_key(p) for p in case["expected_pages"]}
    res["pages_hit"] = sorted(set(res["cited_labels"]) & expected_pages)
    res["must_mention_results"] = check_must_mention(answer, case["must_mention"])
    res["must_not_mention_hits"] = check_must_not_mention(answer, case["must_not_mention"])
    res["flags"] = {
        "no_search": not res["search_queries"],
        "refused_without_search": res["status"] == "not_covered" and not res["search_queries"],
        "selected_other_manual": bool(case.get("manual_id")) and res.get("final_selected_manual") not in (None, case.get("manual_id")),
        "picked_manual_unasked": case.get("manual_id") is None and res.get("final_selected_manual") is not None,
    }
    finalize_checks(case, res)


def finalize_checks(case: dict, res: dict) -> None:
    expected = case["expected_status"]
    checks = {"status": res["status"] == expected}
    checks["pages"] = bool(res["pages_hit"]) if expected == "answered" else None
    checks["must_mention"] = all(m["found"] for m in res["must_mention_results"]) if case["must_mention"] else None
    checks["must_not_mention"] = not res["must_not_mention_hits"] if case["must_not_mention"] else None
    checks["citations_valid"] = (not res["invalid_citations"]) if res["cited_labels"] else None
    judge = res.get("judge") or {}
    checks["faithful"] = judge.get("faithful") if "faithful" in judge else None
    res["checks"] = checks
    res["passed"] = "error" not in res and all(v is not False for v in checks.values())


# ------------------------------------------------------------------ LLM judge


class JudgedClaim(BaseModel):
    claim: str = Field(description="One atomic factual claim from the answer, quoted or closely paraphrased.")
    supported: bool = Field(description="True only if the excerpts explicitly state this claim.")
    evidence: str = Field(description="Label(s) of the excerpt(s) that support the claim, e.g. '[p. 119]', or 'none'.")
    cited: str = Field(description="Page label(s) the answer attaches to this claim, or 'none'.")
    citation_matches: bool = Field(
        description="False only if the claim is cited to a label whose excerpt does not support it; true otherwise."
    )


class UnsupportedClaim(BaseModel):
    claim: str = Field(description="The unsupported claim, quoted or closely paraphrased.")
    reason: str = Field(description="Why the excerpts do not support it (absent, contradicted, adds a cause/number/step...).")


class JudgeVerdict(BaseModel):
    claims: list[JudgedClaim] = Field(description="Every factual claim in the answer, in order.")
    unsupported_claims: list[UnsupportedClaim] = Field(description="Every claim with supported = false.")
    faithful: bool = Field(description="True only if there are no unsupported claims.")


JUDGE_INSTRUCTION = """\
You are an exacting auditor of a retrieval-grounded assistant. The assistant helps motorcycle riders
and is only allowed to use the owner's-manual excerpts it retrieved in this conversation. Your job is
to find every factual claim in its answer that those excerpts do not support.

You receive: the rider's question, whether a photo was attached, the retrieved excerpts (each tagged
with the page label the assistant must cite), and the assistant's answer.

Procedure
1. Split the answer into atomic factual claims: every statement about the motorcycle, its parts,
   specifications, numbers, intervals, procedures, warnings, causes, diagnoses, consequences or
   recommended actions. Split compound statements ("check X and Y" is two claims; a cause plus a
   remedy is two claims).
2. Decide for each claim whether the excerpts support it.
   - SUPPORTED: an excerpt states it explicitly, or the claim is a faithful paraphrase or
     condensation of excerpt wording. Numbers, units, intervals, conditions, part names and the
     link between a symptom and its cause or remedy must match the excerpt.
   - UNSUPPORTED: absent from the excerpts, contradicting them, or going beyond them - an added
     cause or mechanism, an extra step, a number or condition that is not there, a generalisation,
     or a connection the excerpts do not make (for example presenting a maintenance step as the
     explanation of a symptom that the excerpts never link it to).
3. Do NOT use your own knowledge of motorcycles or engines. A claim that is true in the real world
   but not stated in the excerpts is UNSUPPORTED. Do not give the benefit of the doubt.
4. These are not claims - ignore them: the status line at the top of the answer ("From the manual",
   "Not covered by the manual", "Need more information"); questions or requests addressed to the
   rider; statements about what the assistant can or cannot do; the bracketed page labels themselves.
5. A statement that the manual does not cover, describe or mention something is acceptable when the
   excerpts indeed contain nothing on it. If an excerpt does contain that information, the statement
   misrepresents the manual: list it as unsupported.
6. Advice such as "contact an authorised service centre / dealer" is a claim. It is supported if an
   excerpt tells riders to contact or visit the authorised service centre or dealer.
7. If a photo was attached you cannot see it. Plain descriptions of what the photo visibly shows
   (e.g. "white smoke or vapour is coming from the exhaust", "the tail lamp is lit") are allowed: do
   not list them. Any interpretation of the photo - what the smoke or its colour means, a cause, a
   diagnosis, a severity - is a claim that needs support from the excerpts.
8. Citations: record the page label(s) the answer attaches to each claim. If a supported claim is
   cited only to a label whose excerpt does not support it, set citation_matches to false. A claim
   without a citation keeps citation_matches = true (missing citations are checked elsewhere).

Return every claim in `claims`, repeat every unsupported claim in `unsupported_claims` with a short
reason, and set `faithful` to true only if there are no unsupported claims. An answer that contains no
factual claims (for example only a clarification question) is faithful.
"""


def build_judge_prompt(case: dict, res: dict) -> str:
    manual = case.get("manual_id")
    name = retrieval.load_manual(manual).meta["name"] if manual else "none selected"
    out = [
        f"RIDER QUESTION:\n{case['question']}",
        f"\nPHOTO ATTACHED: {'yes' if case.get('attachment') else 'no'}",
        f"\nMANUAL SELECTED: {name}",
        "\nRETRIEVED MANUAL EXCERPTS (the only permitted source):",
    ]
    if not res["excerpts"]:
        out.append("(none - the assistant retrieved no excerpts, so no factual claim about the motorcycle can be supported)")
    for i, ex in enumerate(res["excerpts"], 1):
        out.append(f'<excerpt n="{i}" label="{ex["cite_as"]}" location="{ex["location"]}">\n{ex["text"]}\n</excerpt>')
    out.append(f"\nASSISTANT ANSWER TO AUDIT:\n<answer>\n{res['answer']}\n</answer>")
    return "\n".join(out)


class Judge:
    def __init__(self, primary: str, fallback: str | None, concurrency: int, timeout: float = 300.0):
        self.client = config.genai_client()
        self.models = _unique([m for m in (primary, fallback) if m])
        self.unavailable: dict[str, str] = {}
        self.sem = asyncio.Semaphore(concurrency)
        self.timeout = timeout
        self.calls = 0  # judge requests sent, including retries

    async def __call__(self, case: dict, res: dict) -> dict:
        prompt = build_judge_prompt(case, res)
        cfg = types.GenerateContentConfig(
            system_instruction=JUDGE_INSTRUCTION,
            temperature=0,
            response_mime_type="application/json",
            response_schema=JudgeVerdict,
            labels=config.BILLING_LABELS,
        )
        async with self.sem:
            for model in self.models:
                if model in self.unavailable:
                    continue
                for attempt in range(1, 6):
                    try:
                        self.calls += 1
                        resp = await asyncio.wait_for(
                            self.client.aio.models.generate_content(model=model, contents=prompt, config=cfg),
                            timeout=self.timeout,
                        )
                        verdict = resp.parsed if isinstance(resp.parsed, JudgeVerdict) else JudgeVerdict.model_validate_json(resp.text)
                        return self._summarise(model, resp, verdict)
                    except Exception as exc:  # noqa: BLE001
                        if _model_unavailable(exc):
                            self.unavailable[model] = scrub(f"{type(exc).__name__}: {exc}")[:300]
                            log(f"    judge model {model} unavailable, falling back")
                            break
                        kind = "transient" if isinstance(exc, (asyncio.TimeoutError, TimeoutError)) else classify_error(exc)
                        if kind == "auth" and attempt < 5 and await asyncio.to_thread(refresh_access_token):
                            self.client = config.genai_client()  # rebuild with the new token
                            continue
                        if kind == "transient" and attempt < 5:
                            await _backoff(attempt, f"judge {case['id']}")
                            continue
                        if "validation" in type(exc).__name__.lower() and attempt < 3:
                            continue  # malformed JSON: ask again
                        return {"error": scrub(f"{type(exc).__name__}: {exc}")[:600], "model": model}
            return {"error": f"no judge model available: {self.unavailable}", "model": None}

    @staticmethod
    def _summarise(model: str, resp, verdict: JudgeVerdict) -> dict:
        unsupported = {c.claim.strip(): c.reason for c in verdict.unsupported_claims}
        for c in verdict.claims:  # a claim marked unsupported in `claims` counts even if not repeated
            if not c.supported and c.claim.strip() not in unsupported:
                unsupported[c.claim.strip()] = "marked unsupported in the claim list"
        miscited = [
            {"claim": c.claim, "cited": c.cited, "evidence": c.evidence}
            for c in verdict.claims
            if c.supported and c.cited.strip().lower() not in ("", "none") and not c.citation_matches
        ]
        u = resp.usage_metadata
        usage = {k: int(getattr(u, k)) for k in ("prompt_token_count", "candidates_token_count",
                                                  "thoughts_token_count", "total_token_count")
                 if u is not None and getattr(u, k, None)}
        faithful = not unsupported  # stricter of the two readings
        return {
            "model": model,
            "model_version": getattr(resp, "model_version", None),
            "faithful": faithful,
            "judge_said_faithful": verdict.faithful,
            "inconsistent": verdict.faithful != faithful,
            "n_claims": len(verdict.claims),
            "unsupported_claims": [{"claim": k, "reason": v} for k, v in unsupported.items()],
            "miscited_claims": miscited,
            "claims": [c.model_dump() for c in verdict.claims],
            "usage": usage,
        }


# ------------------------------------------------------------------ metrics


def frac(n: int, d: int) -> str:
    """'n/d (x.x%)' with the percentage truncated, never rounded up."""
    if not d:
        return "n/a"
    return f"{n}/{d} ({math.floor(1000 * n / d) / 10:.1f}%)"


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolation percentile (same as numpy's default)."""
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def compute_metrics(cases: list[dict], results: dict[str, dict]) -> dict:
    rows = [(c, results[c["id"]]) for c in cases if c["id"] in results]
    m: dict = {"n_cases": len(rows)}

    def rate(pairs):
        return {"n": sum(pairs), "d": len(pairs), "text": frac(sum(pairs), len(pairs))}

    m["status_accuracy"] = rate([r["status"] == c["expected_status"] for c, r in rows])
    m["status_accuracy_by_class"] = {
        s: rate([r["status"] == s for c, r in rows if c["expected_status"] == s]) for s in STATUSES
    }
    m["status_accuracy_guardrail_as_not_covered"] = rate(
        [(("not_covered" if r["status"] == "guardrail" else r["status"]) == c["expected_status"]) for c, r in rows]
    )
    actual_cols = ["answered", "not_covered", "need_info", "guardrail", "unparsed", "empty", "error"]
    m["confusion"] = {
        s: {a: sum(1 for c, r in rows if c["expected_status"] == s and r["status"] == a) for a in actual_cols}
        for s in STATUSES
    }
    answered = [(c, r) for c, r in rows if c["expected_status"] == "answered"]
    m["expected_page_hit"] = rate([bool(r.get("pages_hit")) for c, r in answered])
    mm = [(c, r) for c, r in rows if c["must_mention"]]
    m["must_mention_pass"] = rate([all(x["found"] for x in r.get("must_mention_results", [])) for c, r in mm])
    items = [x["found"] for c, r in mm for x in r.get("must_mention_results", [])]
    m["must_mention_items"] = rate(items)
    mnm = [(c, r) for c, r in rows if c["must_not_mention"]]
    m["must_not_mention_violations"] = {
        "cases": sum(1 for c, r in mnm if r.get("must_not_mention_hits")),
        "of": len(mnm),
        "hits": [{"id": c["id"], "terms": [h["term"] for h in r["must_not_mention_hits"]]}
                 for c, r in mnm if r.get("must_not_mention_hits")],
    }
    cited = [(c, r) for c, r in rows if r.get("cited_labels")]
    m["citation_validity"] = rate([not r["invalid_citations"] for c, r in cited])
    m["guardrail_triggers"] = [c["id"] for c, r in rows if r["status"] == "guardrail"]
    lat = [r["latency_s"] for c, r in rows if "latency_s" in r]
    m["latency_s"] = {"median": statistics.median(lat) if lat else None, "p90": percentile(lat, 0.9),
                      "max": max(lat) if lat else None, "n": len(lat)}
    model_s = [r["model_seconds"] for c, r in rows if "model_seconds" in r]
    tool_s = [r["tool_seconds"] for c, r in rows if "tool_seconds" in r]
    m["model_seconds_median"] = statistics.median(model_s) if model_s else None
    m["tool_seconds_median"] = statistics.median(tool_s) if tool_s else None
    judged = [(c, r) for c, r in rows if "faithful" in (r.get("judge") or {})]
    m["faithfulness"] = rate([r["judge"]["faithful"] for c, r in judged])
    m["judge_errors"] = [c["id"] for c, r in rows if (r.get("judge") or {}).get("error")]
    m["unsupported_claims_total"] = sum(len(r["judge"]["unsupported_claims"]) for c, r in judged)
    m["miscited_claims_total"] = sum(len(r["judge"]["miscited_claims"]) for c, r in judged)
    m["judge_inconsistent"] = [c["id"] for c, r in judged if r["judge"].get("inconsistent")]
    m["passed"] = rate([r["passed"] for c, r in rows])
    m["errors"] = [c["id"] for c, r in rows if "error" in r]
    m["no_search"] = [c["id"] for c, r in rows if r.get("flags", {}).get("no_search") and "error" not in r]
    m["refused_without_search"] = [c["id"] for c, r in rows if r.get("flags", {}).get("refused_without_search")]
    m["status_line_not_exact"] = [c["id"] for c, r in rows
                                  if r["status"] in STATUSES and not r.get("status_line_exact")]
    m["status_line_moved"] = [c["id"] for c, r in rows if r.get("reply_rewritten") == "status_line_moved"]
    m["searches_per_case"] = (statistics.mean(len(r.get("search_queries", [])) for c, r in rows) if rows else None)

    def tok(key, source):
        return sum((source(r) or {}).get(key, 0) for c, r in rows)

    m["agent_tokens"] = {k: tok(k, lambda r: r.get("usage")) for k in (
        "prompt_token_count", "candidates_token_count", "thoughts_token_count", "total_token_count")}
    m["agent_model_calls"] = sum(r.get("n_model_calls", 0) for c, r in rows)
    m["judge_tokens"] = {k: tok(k, lambda r: (r.get("judge") or {}).get("usage")) for k in (
        "prompt_token_count", "candidates_token_count", "thoughts_token_count", "total_token_count")}
    return m


# ------------------------------------------------------------------ outputs


def _public(res: dict) -> dict:
    """Result without manual excerpt text (copyrighted) - excerpts keep only their labels."""
    out = {k: v for k, v in res.items() if k not in ("excerpts", "searches", "model_calls")}
    out["excerpts"] = [{k: v for k, v in ex.items() if k != "text"} for ex in res.get("excerpts", [])]
    out["model_calls"] = [{k: v for k, v in mc.items() if k != "text"} for mc in res.get("model_calls", [])]
    return out


def _raw(res: dict) -> dict:
    return {k: v for k, v in res.items() if k != "searches"}


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False, default=str), encoding="utf-8")


def _yn(v) -> str:
    return "-" if v is None else ("yes" if v else "**no**")


def _judge_cell(res: dict) -> str:
    j = res.get("judge")
    if not j:
        return "-"
    if j.get("error"):
        return "error"
    n = len(j["unsupported_claims"])
    return "faithful" if j["faithful"] else f"**unfaithful ({n})**"


def _short(text: str, limit: int = 420) -> str:
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    if lines and STATUS_RE.match(lines[0]):
        lines = lines[1:]
    body = " ".join(ln.strip() for ln in lines)
    return body if len(body) <= limit else body[:limit].rsplit(" ", 1)[0] + " ..."


def _md_escape(text: str) -> str:
    return (text or "").replace("|", "\\|").replace("\n", " ")


def load_comparison(path: Path, cases: list[dict]) -> dict:
    """An earlier run's stored answers, re-scored against the current cases (judge verdicts kept)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    by_id = {c["id"]: c for c in cases}
    results = {}
    for r in data["cases"]:
        if r.get("id") in by_id:
            r = json.loads(json.dumps(r))  # private copy
            score(by_id[r["id"]], r)
            results[r["id"]] = r
    run = dict(data.get("run") or {})
    run["agent_model"] = _clean_model_name(run.get("agent_model"))
    return {"path": str(path), "run": run, "results": results, "metrics": compute_metrics(cases, results)}


def comparison_section(run: dict, cases: list[dict], results: dict, metrics: dict, comp: dict) -> list[str]:
    old, om = comp["results"], comp["metrics"]
    rows = [
        ("Status accuracy", "status_accuracy"), ("Expected-page hit rate", "expected_page_hit"),
        ("must_mention pass rate", "must_mention_pass"), ("Citation validity", "citation_validity"),
        ("Faithfulness (judge)", "faithfulness"), ("Cases passing every check", "passed"),
    ]
    merged = comp["run"].get("merged_runs") or []
    L = [
        f"## Previous run (`{comp['run'].get('run_id')}`) vs this run",
        "",
        f"Earlier run `{comp['run'].get('run_id')}` ({comp['run'].get('started_at')} UTC, agent "
        f"`{comp['run'].get('agent_model')}`"
        + (f", with targeted re-runs {', '.join(f'`{x}`' for x in merged)} merged in" if merged else "")
        + "), re-scored against the current `cases.json`. The agent code may differ between the two runs, "
        "and each run is a single sample per case.",
        "",
        f"| Metric | Earlier `{comp['run'].get('run_id')}` | This run `{run['run_id']}` |",
        "|---|---|---|",
    ]
    for label, key in rows:
        L.append(f"| {label} | {om[key]['text']} | {metrics[key]['text']} |")
    ov, nv = om["must_not_mention_violations"], metrics["must_not_mention_violations"]
    L.append(f"| must_not_mention violations | {ov['cases']} of {ov['of']} | {nv['cases']} of {nv['of']} |")
    L.append(f"| Guardrail triggers | {len(om['guardrail_triggers'])} | {len(metrics['guardrail_triggers'])} |")
    lo, ln = om["latency_s"], metrics["latency_s"]
    if lo["median"] is not None and ln["median"] is not None:
        L.append(f"| Latency median / p90 | {lo['median']:.1f} s / {lo['p90']:.1f} s | {ln['median']:.1f} s / {ln['p90']:.1f} s |")
    changed = []
    for c in cases:
        a, b = old.get(c["id"]), results.get(c["id"])
        if a and b and (a["passed"] != b["passed"] or a["status"] != b["status"]):
            changed.append((c, a, b))
    L += ["", f"Cases whose status or pass/fail changed: {len(changed)}.", ""]
    if changed:
        L += ["| id | earlier status | earlier result | this run status | this run result |", "|---|---|---|---|---|"]
        for c, a, b in changed:
            L.append(f"| {c['id']} | {a['status']} | {'pass' if a['passed'] else '**fail**'} | {b['status']} | "
                     f"{'pass' if b['passed'] else '**fail**'} |")
    stable_fail = [c["id"] for c in cases if old.get(c["id"]) and results.get(c["id"])
                   and not old[c["id"]]["passed"] and not results[c["id"]]["passed"]]
    if stable_fail:
        L += ["", "Failing in both runs: " + ", ".join(stable_fail) + "."]
    L.append("")
    return L


def write_report(path: Path, run: dict, cases: list[dict], results: dict[str, dict], metrics: dict,
                 comparison: dict | None = None) -> None:
    preserved = None
    if path.exists():
        old = path.read_text(encoding="utf-8")
        if ANALYSIS_START in old and ANALYSIS_END in old:
            preserved = old.split(ANALYSIS_START, 1)[1].split(ANALYSIS_END, 1)[0].strip("\n")
    m = metrics
    by = m["status_accuracy_by_class"]
    lat = m["latency_s"]
    at, jt = m["agent_tokens"], m["judge_tokens"]
    mnm = m["must_not_mention_violations"]
    judge_models = ", ".join(run.get("judge_models_used") or []) or "not run"
    L = [
        "# Evaluation results",
        "",
        "Generated by `evals/run_eval.py` from `evals/cases.json`. "
        "Do not edit by hand except between the analysis markers (that block is preserved on regeneration).",
        "",
        "| | |",
        "|---|---|",
        f"| Run id | `{run['run_id']}` |",
        f"| Date (UTC) | {run['started_at']} to {run['finished_at']} |",
        f"| Cases | {m['n_cases']} ({', '.join(f'{sum(1 for c in cases if c['id'] in results and c['expected_status'] == s)} {s}' for s in STATUSES)}) |",
        f"| Agent model | `{run['agent_model']}` (served as: {', '.join(f'`{v}`' for v in run.get('agent_model_versions') or []) or 'n/a'}), temperature 0.1 |",
        f"| Retrieval | `{run['embedding_model']}` embeddings + BM25 (RRF), top_k = {run['top_k']} |",
        f"| Judge model | {judge_models} (requested `{run['judge_model_requested']}`, fallback `{run['judge_fallback']}`), temperature 0 |",
        f"| Concurrency | {run['concurrency']} cases at a time; fresh Runner, session and event loop per case |",
        f"| Agent questions run | {m['n_cases']} in this report ({m['agent_model_calls']} agent model calls; "
        f"{run.get('agent_attempts', '?')} attempts incl. retries after transient errors; "
        f"{run.get('judge_requests', '?')} judge requests) |",
        "",
        "## Summary",
        "",
        "| Metric | Result |",
        "|---|---|",
        f"| **Status accuracy (all cases)** | {m['status_accuracy']['text']} |",
        f"| - expected *answered* | {by['answered']['text']} |",
        f"| - expected *not_covered* | {by['not_covered']['text']} |",
        f"| - expected *need_info* | {by['need_info']['text']} |",
        f"| Status accuracy if guardrail replies counted as *not_covered* | {m['status_accuracy_guardrail_as_not_covered']['text']} |",
        f"| Expected-page hit rate (answered cases; >= 1 expected page cited) | {m['expected_page_hit']['text']} |",
        f"| must_mention pass rate (cases; all facts present) | {m['must_mention_pass']['text']} |",
        f"| must_mention facts found (individual items) | {m['must_mention_items']['text']} |",
        f"| must_not_mention violations (cases with >= 1 forbidden term) | {mnm['cases']} of {mnm['of']} |",
        f"| Citation validity (answers whose cited pages were all retrieved) | {m['citation_validity']['text']} |",
        f"| Guardrail triggers | {len(m['guardrail_triggers'])}{(' (' + ', '.join(m['guardrail_triggers']) + ')') if m['guardrail_triggers'] else ''} |",
        f"| **Faithfulness (LLM judge: no unsupported claims)** | {m['faithfulness']['text']} |",
        f"| Unsupported claims found by the judge (total) | {m['unsupported_claims_total']} |",
        f"| Supported claims cited to the wrong page (judge) | {m['miscited_claims_total']} |",
        f"| **Cases passing every check** | {m['passed']['text']} |",
        f"| Latency per question: median / p90 / max | {lat['median']:.1f} s / {lat['p90']:.1f} s / {lat['max']:.1f} s |" if lat["median"] is not None else "| Latency | n/a |",
        f"| - median time inside model calls / inside tool calls | {m['model_seconds_median']:.1f} s / {m['tool_seconds_median']:.1f} s |" if m.get("model_seconds_median") is not None else "| - latency breakdown | n/a |",
        f"| Agent tokens (prompt / output / thinking / total) | {at['prompt_token_count']:,} / {at['candidates_token_count']:,} / {at['thoughts_token_count']:,} / {at['total_token_count']:,} |",
        f"| Judge tokens (prompt / output / thinking / total) | {jt['prompt_token_count']:,} / {jt['candidates_token_count']:,} / {jt['thoughts_token_count']:,} / {jt['total_token_count']:,} |",
        "",
        "Percentages are truncated to one decimal (never rounded up). A case *passes* when its status matches and "
        "every applicable check holds: expected page cited and all must_mention facts present (answered cases), "
        "no must_not_mention term, every cited page actually retrieved, and the judge found no unsupported claim. "
        "p90 uses linear interpolation. Latency is wall-clock per question including tool calls and embeddings, "
        f"measured with {run['concurrency']} questions in flight, each on its own event loop (the agent's "
        "synchronous `search_manual` tool blocks the loop it runs on during its embedding call).",
        "",
        "Other observations: "
        + "; ".join(x for x in [
            f"errors after retries: {', '.join(m['errors'])}" if m["errors"] else "no case errored after retries",
            f"answered/refused without any manual search: {', '.join(m['no_search'])}" if m["no_search"] else "every case searched the manual at least once",
            f"status line not in the exact required form: {', '.join(m['status_line_not_exact'])}" if m["status_line_not_exact"] else "all status lines in the exact required form",
            f"model put the status line below the first line and the agent's `status_first` moved it up: {', '.join(m['status_line_moved'])}" if m.get("status_line_moved") else "no reply needed its status line moved to the top",
            f"judge errors: {', '.join(m['judge_errors'])}" if m["judge_errors"] else None,
            f"mean searches per question: {m['searches_per_case']:.1f}" if m["searches_per_case"] is not None else None,
        ] if x) + ".",
        "",
        "### Status confusion matrix (rows: expected, columns: actual)",
        "",
        "| expected \\ actual | answered | not_covered | need_info | guardrail | unparsed/empty/error |",
        "|---|---|---|---|---|---|",
    ]
    extra = []
    if run.get("merged_runs"):
        extra.append(f"| Targeted re-runs merged in | {', '.join(f'`{x}`' for x in run['merged_runs'])} (see the run column) |")
    if run.get("rescored"):
        extra.append(f"| Rescored | {run['rescored']['at']} UTC: deterministic checks recomputed on the stored answers "
                     f"after a cases.json change (no new model calls; judge verdicts unchanged) |")
    at_row = next(i for i, line in enumerate(L) if line.startswith("| Agent questions run"))
    L[at_row + 1:at_row + 1] = extra
    for s in STATUSES:
        row = m["confusion"][s]
        L.append(f"| {s} | {row['answered']} | {row['not_covered']} | {row['need_info']} | {row['guardrail']} | "
                 f"{row['unparsed'] + row['empty'] + row['error']} |")
    L += [
        "",
        "## Per-case results",
        "",
        "| id | manual | expected | actual | pages hit | must_mention | must_not | cites retrieved | judge | latency s | run |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in cases:
        r = results.get(c["id"])
        if not r:
            continue
        ch = r["checks"]
        pages = ", ".join(p.replace("p.", "p. ") for p in r["pages_hit"]) if r["pages_hit"] else ("**none**" if ch["pages"] is False else "-")
        actual = r["status"] if r["status"] == c["expected_status"] else f"**{r['status']}**"
        L.append(
            f"| {c['id']} | {SHORT_MANUAL.get(c.get('manual_id'), 'none selected')} | {c['expected_status']} | {actual} | "
            f"{pages} | {_yn(ch['must_mention'])} | {_yn(ch['must_not_mention'])} | {_yn(ch['citations_valid'])} | "
            f"{_judge_cell(r)} | {r.get('latency_s', '-')} | {'' if r.get('run_id') == run['run_id'] else r.get('run_id', '')} |"
        )
    failing = [(c, results[c["id"]]) for c in cases if c["id"] in results and not results[c["id"]]["passed"]]
    L += ["", "## Failures and analysis", "", f"{len(failing)} of {len(results)} cases failed at least one check.", ""]
    for c, r in failing:
        ch = r["checks"]
        why = []
        if not ch["status"]:
            why.append(f"status {r['status']} (expected {c['expected_status']})")
        if ch["pages"] is False:
            why.append(f"no expected page cited (expected {', '.join(c['expected_pages'])}; cited {', '.join(r['cited_labels']) or 'none'})")
        if ch["must_mention"] is False:
            missing = [" / ".join(x["item"]) for x in r["must_mention_results"] if not x["found"]]
            why.append("missing fact(s): " + "; ".join(f"`{x}`" for x in missing))
        if ch["must_not_mention"] is False:
            why.append("forbidden term(s): " + ", ".join(f"`{h['term']}`" for h in r["must_not_mention_hits"]))
        if ch["citations_valid"] is False:
            why.append("cited pages never retrieved: " + ", ".join(r["invalid_citations"]))
        if ch["faithful"] is False:
            why.append(f"judge: {len(r['judge']['unsupported_claims'])} unsupported claim(s)")
        if "error" in r:
            why.append(f"error: {r['error'][:200]}")
        L.append(f"**{c['id']}** ({SHORT_MANUAL.get(c.get('manual_id'), 'none selected')}) - {'; '.join(why)}")
        L.append("")
        L.append(f"- Question: {c['question']}" + (f" (photo: `{c['attachment']}`)" if c.get("attachment") else ""))
        L.append(f"- Searches: {'; '.join(f'`{q}`' for q in r.get('search_queries', [])) or '(none)'}")
        L.append(f"- Answer ({r['status']}): {_short(r.get('answer', ''))}")
        if r.get("raw_model_answer"):
            how = ("before the guardrail replaced it" if r.get("reply_rewritten") != "status_line_moved"
                   else "before `status_first` moved its status line to the top")
            L.append(f"- Model's reply {how} ({r.get('raw_status')}; cited "
                     f"{', '.join(r.get('raw_cited_labels') or []) or 'nothing'}; not retrieved: "
                     f"{', '.join(r.get('raw_invalid_citations') or []) or 'none'}): {_short(r['raw_model_answer'])}")
        for u in (r.get("judge") or {}).get("unsupported_claims", []):
            L.append(f"- Judge, unsupported: \"{u['claim']}\" - {u['reason']}")
        for h in r.get("must_not_mention_hits", []):
            L.append(f"- Forbidden term `{h['term']}` in: \"...{h['context']}...\"")
        L.append("")
    miscited = [(c, results[c["id"]]) for c in cases if c["id"] in results
                and (results[c["id"]].get("judge") or {}).get("miscited_claims")]
    if miscited:
        L += ["Supported claims the judge found cited to the wrong page (not counted as failures):", ""]
        for c, r in miscited:
            for x in r["judge"]["miscited_claims"]:
                L.append(f"- {c['id']}: \"{x['claim']}\" cited {x['cited']}, supported by {x['evidence']}")
        L.append("")
    if comparison:
        L += comparison_section(run, cases, results, metrics, comparison)
    L += [ANALYSIS_START, preserved if preserved is not None else "_Analysis of root causes: pending._", ANALYSIS_END, ""]
    L += [
        "## Known limitations of this harness",
        "",
        "- One sample per case at the agent's temperature (0.1): results are not deterministic, so a single pass or "
        "fail is noisy. With 32 cases each case moves a rate by about 3 percentage points.",
        "- must_mention / must_not_mention are lexical checks. They can miss a correct paraphrase or count a negated "
        "mention (\"this is not a head gasket issue\") as a violation, so every hit is read by hand in the analysis.",
        "- Expected pages are judgement calls made from the indexed text; the hit check only requires one of them.",
        "- Citation validity compares page labels, not chunks: a cited page counts as valid if any chunk with that "
        "label was retrieved, even if the fact came from a different chunk on that page.",
        "- The faithfulness judge is itself an LLM (from the same model family as the agent). It sees only the "
        "retrieved excerpts, not the full manual or the photo, and can make mistakes in either direction; its "
        "verdicts on failing cases were read by hand, the passing ones were not all audited.",
        "- Token counts come from the agent's model responses (captured by a Runner plugin before the guardrail) "
        "and from the judge responses; embedding calls made by `search_manual` are not counted.",
        "- Clarification cases only check the first turn (the agent asks which bike), not the follow-up turn.",
        "- The photo cases use a single AI-generated image.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "export GOOGLE_GENAI_USE_VERTEXAI=TRUE GOOGLE_CLOUD_PROJECT=<project> GOOGLE_CLOUD_LOCATION=global",
        "export GOOGLE_OAUTH_ACCESS_TOKEN=$(gcloud auth print-access-token)   # or use ADC",
        "python evals/run_eval.py                 # full run with judge",
        "python evals/run_eval.py --only <ids>    # targeted re-run (add --merge to update this report)",
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")


# ------------------------------------------------------------------ main


async def run_all(args) -> None:
    cases = load_cases(args.cases)
    if args.only:
        wanted = [i for chunk in args.only for i in chunk.split(",") if i]
        unknown = sorted(set(wanted) - {c["id"] for c in cases})
        if unknown:
            raise SystemExit(f"Unknown case id(s): {', '.join(unknown)}")
        selected = [c for c in cases if c["id"] in set(wanted)]
    else:
        selected = cases
    now = dt.datetime.now(dt.timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%SZ") + ("-only" if args.only else "")
    judge = None if args.no_judge else Judge(
        args.judge_model, args.judge_fallback, args.judge_concurrency, args.judge_timeout
    )
    log(f"Run {run_id}: {len(selected)} case(s), agent {agent_model_name()}, "
        f"judge {'off' if judge is None else args.judge_model}, concurrency {args.concurrency}")
    sem = asyncio.Semaphore(args.concurrency)
    results: dict[str, dict] = {}

    async def one(case: dict) -> None:
        async with sem:
            res = await run_agent_with_retry(case, args.retries, args.case_timeout)
        res["run_id"] = run_id
        res.setdefault("answer", "")
        evaluate(case, res)
        if judge is not None and "error" not in res:
            res["judge"] = await judge(case, res)
            finalize_checks(case, res)
        results[case["id"]] = res
        j = res.get("judge") or {}
        verdict = "-" if not j else ("error" if j.get("error") else ("faithful" if j["faithful"] else f"UNFAITHFUL({len(j['unsupported_claims'])})"))
        log(f"  {'PASS' if res['passed'] else 'FAIL'} {case['id']:<32} expected {case['expected_status']:<11} "
            f"got {res['status']:<11} {res.get('latency_s', 0):>6.1f}s searches={len(res.get('search_queries', []))} "
            f"judge={verdict}" + (f"  error={res['error'][:120]}" if "error" in res else ""))

    await asyncio.gather(*(one(c) for c in selected))
    finished = dt.datetime.now(dt.timezone.utc)
    versions = _unique(v for r in results.values() for v in r.get("model_versions", []))
    judge_models = _unique((r.get("judge") or {}).get("model") for r in results.values() if (r.get("judge") or {}).get("model"))
    run = {
        "run_id": run_id,
        "started_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "finished_at": finished.strftime("%Y-%m-%d %H:%M:%S"),
        "argv": scrub(" ".join(sys.argv[1:])),
        "agent_model": agent_model_name(),
        "agent_model_versions": versions,
        "embedding_model": config.EMBEDDING_MODEL,
        "top_k": config.TOP_K,
        "judge_model_requested": None if args.no_judge else args.judge_model,
        "judge_fallback": None if args.no_judge else args.judge_fallback,
        "judge_models_used": judge_models,
        "judge_unavailable": judge.unavailable if judge else {},
        "concurrency": args.concurrency,
        "agent_attempts": sum(r.get("attempts", 0) for r in results.values()),
        "judge_requests": judge.calls if judge else 0,
        "cases_file": str(args.cases.relative_to(ROOT)) if args.cases.is_relative_to(ROOT) else str(args.cases),
        "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
    }
    ordered = {c["id"]: results[c["id"]] for c in selected if c["id"] in results}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(RESULTS_DIR / f"{run_id}.raw.json", {"run": run, "cases": [_raw(r) | {"id": k} for k, r in ordered.items()]})

    if args.only and not args.merge:
        m = compute_metrics(selected, ordered)
        log(f"\nTargeted run written to {RESULTS_DIR / (run_id + '.raw.json')} (latest.json and the report untouched).")
        log(f"Status {m['status_accuracy']['text']}, passed {m['passed']['text']}, faithful {m['faithfulness']['text']}")
        return

    merged, runs = dict(ordered), [run]
    latest_raw = RESULTS_DIR / "latest.raw.json"
    if args.only and args.merge:
        if not latest_raw.exists():
            raise SystemExit("--merge needs an existing evals/results/latest.raw.json from a full run")
        prev = json.loads(latest_raw.read_text(encoding="utf-8"))
        merged = {r["id"]: r for r in prev["cases"]}
        merged.update({k: r | {"id": k} for k, r in ordered.items()})
        runs = prev.get("runs", [prev.get("run")]) + [run]
        base = prev.get("run") or run
        run = dict(base, merged_runs=[x["run_id"] for x in runs[1:]], finished_at=run["finished_at"])
        run["agent_model"] = _clean_model_name(run.get("agent_model"))
        run["judge_models_used"] = _unique((base.get("judge_models_used") or []) + judge_models)
        run["agent_model_versions"] = _unique((base.get("agent_model_versions") or []) + versions)
        run["agent_attempts"] = sum(x.get("agent_attempts", 0) for x in runs)  # across the merged runs
        run["judge_requests"] = sum(x.get("judge_requests", 0) for x in runs)
    merged = {c["id"]: merged[c["id"]] | {"id": c["id"]} for c in cases if c["id"] in merged}
    metrics = compute_metrics(cases, merged)
    comparison = _comparison_for(args, run, cases)
    _write_outputs(args, run, runs, cases, merged, metrics, comparison)
    log(f"\nWrote {RESULTS_DIR / 'latest.json'} and {args.report}")
    log(f"Status {metrics['status_accuracy']['text']} | pages {metrics['expected_page_hit']['text']} | "
        f"must_mention {metrics['must_mention_pass']['text']} | violations {metrics['must_not_mention_violations']['cases']} | "
        f"citations {metrics['citation_validity']['text']} | guardrail {len(metrics['guardrail_triggers'])} | "
        f"faithful {metrics['faithfulness']['text']} | passed {metrics['passed']['text']}")


def rescore_latest(args) -> None:
    """Recompute the deterministic checks of the stored answers (no model calls) against the
    current cases.json, keep the judge verdicts, and rewrite latest.json and the report."""
    cases = load_cases(args.cases)
    raw_path = RESULTS_DIR / "latest.raw.json"
    prev = json.loads(raw_path.read_text(encoding="utf-8"))
    by_id = {c["id"]: c for c in cases}
    stored = {r["id"]: r for r in prev["cases"] if r["id"] in by_id}
    for cid, r in stored.items():
        score(by_id[cid], r)
    merged = {c["id"]: stored[c["id"]] for c in cases if c["id"] in stored}
    run = dict(prev["run"])
    run["agent_model"] = _clean_model_name(run.get("agent_model"))
    sha = hashlib.sha256(args.cases.read_bytes()).hexdigest()
    if sha != run.get("cases_sha256"):
        run["rescored"] = {"at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                           "cases_sha256": sha, "previous_cases_sha256": run.get("cases_sha256")}
        run["cases_sha256"] = sha
    metrics = compute_metrics(cases, merged)
    runs = prev.get("runs") or [run]
    comparison = _comparison_for(args, run, cases)
    _write_outputs(args, run, runs, cases, merged, metrics, comparison)
    log(f"Rescored {len(merged)} stored results; wrote {RESULTS_DIR / 'latest.json'} and {args.report}")


def _comparison_for(args, run: dict, cases: list[dict]) -> dict | None:
    """--compare PATH (remembered in the run record so later --merge/--rescore keep it)."""
    path = args.compare or (ROOT / run["compare_path"] if run.get("compare_path") else None)
    if not path:
        return None
    path = Path(path).resolve()
    if not path.exists():
        log(f"Comparison file {path} not found - skipping the comparison section")
        return None
    comparison = load_comparison(path, cases)
    run["compare_path"] = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    run["compared_with"] = comparison["run"].get("run_id")
    return comparison


def _write_outputs(args, run, runs, cases, merged, metrics, comparison) -> None:
    extra = {}
    if comparison:
        extra["comparison"] = {"run_id": comparison["run"].get("run_id"), "metrics": comparison["metrics"]}
    raw_path = RESULTS_DIR / "latest.raw.json"
    _write_json(raw_path, {"run": run, "runs": runs, "metrics": metrics, **extra,
                           "cases": [_raw(r) for r in merged.values()]})
    _write_json(RESULTS_DIR / "latest.json", {"run": run, "runs": runs, "metrics": metrics, **extra,
                                               "cases": [_public(r) for r in merged.values()]})
    write_report(args.report, run, cases, merged, metrics, comparison)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", type=Path, default=CASES_FILE)
    parser.add_argument("--only", nargs="+", metavar="ID", help="Run only these case ids (space or comma separated)")
    parser.add_argument("--merge", action="store_true", help="With --only: replace those cases in latest.json and regenerate the report")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--no-judge", action="store_true", help="Skip the LLM faithfulness judge")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE)
    parser.add_argument("--judge-fallback", default=FALLBACK_JUDGE)
    parser.add_argument("--judge-concurrency", type=int, default=4)
    parser.add_argument("--retries", type=int, default=5, help="Retries per case on transient 429/5xx/network errors")
    parser.add_argument("--case-timeout", type=float, default=420.0, help="Seconds before a hung case is retried")
    parser.add_argument("--judge-timeout", type=float, default=300.0, help="Seconds before a hung judge call is retried")
    parser.add_argument("--report", type=Path, default=REPORT_FILE)
    parser.add_argument("--compare", type=Path, metavar="RAW_JSON",
                        help="Add a comparison with an earlier run (a *.raw.json written by this script)")
    parser.add_argument("--rescore", action="store_true",
                        help="No model calls: re-check the stored answers in latest.raw.json against cases.json "
                             "and regenerate latest.json and the report")
    args = parser.parse_args()
    args.cases = args.cases.resolve()
    if args.rescore:
        rescore_latest(args)
        return
    asyncio.run(run_all(args))
    if ABORT["reason"]:
        log(f"\nRun aborted early: {ABORT['reason']}")
        sys.exit(2)


if __name__ == "__main__":
    main()
