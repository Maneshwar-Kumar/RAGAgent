"""ADK agent: grounded bike-manual troubleshooting.

Grounding is enforced in three layers:
1. Retrieval: the model only sees manual text through the `search_manual` tool.
2. Instruction: answers must come from retrieved excerpts, with a page citation per fact,
   and must say "Not covered by the manual" otherwise (see prompts.py).
3. Guardrail (`enforce_citations`): a deterministic check after every model reply. A reply
   that claims to answer from the manual but has no citations, cites a page that was never
   retrieved from the selected manual in this session, or is a long reply with neither a status
   line nor citations, is replaced with a "not covered" message.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.google_llm import Gemini
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools import ToolContext
from google.genai import types

from . import config, retrieval
from .ingest import ingest
from .prompts import INSTRUCTION

log = logging.getLogger(__name__)

SELECTED = "selected_manual"
CITABLE = "citable_labels"  # manual_id -> page labels returned by search_manual this session
UPLOADS = "uploaded_manuals"  # sha256 -> manual_id
FAILED_UPLOADS = "failed_uploads"  # sha256 -> reason the PDF could not be indexed

STATUS_LINES = {
    "from the manual": "answered",
    "not covered by the manual": "not_covered",
    "need more information": "need_info",
}
UNCITED_REPLY_LIMIT = 400  # longer replies without status line or citation look like answers
_LABEL_RE = re.compile(r"(PDF p\.|p\.)\s*([A-Za-z0-9\-\u2013]+)")
_BRACKET_RE = re.compile(r"\[([^\]]*\bp\.[^\]]*)\]")
_RANGE_RE = re.compile(r"^(\d+)[\-\u2013](\d+)$")


UPLOAD_PREFIX = "upload-"


def visible_manual_ids(state) -> list[str]:
    """Pre-loaded manuals plus manuals uploaded in this session (uploads stay private)."""
    own_uploads = set(state.get(UPLOADS, {}).values())
    return [
        mid
        for mid in retrieval.list_manual_ids()
        if not mid.startswith(UPLOAD_PREFIX) or mid in own_uploads
    ]


def _manual_names(state) -> dict[str, str]:
    return {mid: retrieval.load_manual(mid).meta["name"] for mid in visible_manual_ids(state)}


def _current_manual(state) -> str | None:
    ids = visible_manual_ids(state)
    selected = state.get(SELECTED)
    if selected in ids:
        return selected
    return ids[0] if len(ids) == 1 else None


# ---------------------------------------------------------------- tools


def list_manuals(tool_context: ToolContext) -> dict:
    """List the bike manuals that have been loaded and which one is selected."""
    return {
        "manuals": [
            {"manual_id": k, "name": v} for k, v in _manual_names(tool_context.state).items()
        ],
        "selected": _current_manual(tool_context.state),
    }


def select_manual(manual_id: str, tool_context: ToolContext) -> dict:
    """Select which bike's manual to use for the rest of the conversation.

    Args:
        manual_id: The id from list_manuals.
    """
    if manual_id not in visible_manual_ids(tool_context.state):
        return {"error": f"Unknown manual_id '{manual_id}'.", **list_manuals(tool_context)}
    tool_context.state[SELECTED] = manual_id
    return {"selected": manual_id, "name": retrieval.load_manual(manual_id).meta["name"]}


async def search_manual(query: str, tool_context: ToolContext) -> dict:
    """Search the selected bike manual and return the most relevant excerpts with page labels.

    Args:
        query: The symptom, component or procedure to look up, phrased the way a manual would
            describe it (e.g. "engine does not start", "coolant level check").
    """
    manual_id = _current_manual(tool_context.state)
    if manual_id is None:
        return {
            "error": "No manual selected. Ask the rider which bike they have, then call "
            "select_manual.",
            **list_manuals(tool_context),
        }
    manual = retrieval.load_manual(manual_id)
    # Embedding the query is network I/O: run it off the event loop so that other riders'
    # turns keep moving while this one waits.
    result = await asyncio.to_thread(retrieval.search, manual, query)
    tool_context.state[SELECTED] = manual_id  # record an implicit single-manual choice
    citable = dict(tool_context.state.get(CITABLE, {}))
    labels = set(citable.get(manual_id, []))
    labels.update(_normalise_labels(r["cite_as"]) for r in result["results"])
    citable[manual_id] = sorted(labels)
    tool_context.state[CITABLE] = citable
    return result


# ---------------------------------------------------------------- callbacks


def _normalise_labels(text: str) -> str:
    kind, page = _LABEL_RE.search(text).groups()
    return f"{kind.replace(' ', '')}{page}".lower()


def _cited_labels(text: str) -> list[str]:
    """Pages cited in brackets: "[p. 66, 75]" is p. 66 and p. 75; "[p. 66-67]" both pages."""
    found = []
    for group in _BRACKET_RE.findall(text):
        kind = "p."
        for token in re.split(r"[,;]|\band\b", group):
            match = _LABEL_RE.search(token)
            if match:
                kind, page = match.groups()
            elif bare := re.search(r"\b(\d+(?:[\-\u2013]\d+)?)\b", token):
                page = bare.group(1)
            else:
                continue
            prefix = kind.replace(" ", "").lower()
            span = _RANGE_RE.match(page)
            if span and 0 < int(span[2]) - int(span[1]) <= 5:
                found += [f"{prefix}{n}" for n in range(int(span[1]), int(span[2]) + 1)]
            else:
                found.append(f"{prefix}{page}".lower())
    return found


def reply_status(text: str) -> str | None:
    """Status from the reply's first line ("**From the manual**" -> "answered"), if any."""
    first = text.strip().split("\n", 1)[0]
    key = re.sub(r"[*_#>`]", "", first).strip().rstrip(".:!").strip().lower()
    return STATUS_LINES.get(key)


def status_first(text: str) -> str:
    """Move a status line that the model put later in the reply (e.g. after a photo
    description) to the top, where the rider UI and the guardrail expect it."""
    lines = text.strip().split("\n")
    if reply_status(lines[0]):
        return text
    for i, line in enumerate(lines[1:], start=1):
        if reply_status(line):
            rest = lines[:i] + lines[i + 1 :]
            return "\n".join([line.strip(), ""] + rest).strip()
    return text


def _fallback(reason: str) -> LlmResponse:
    msg = (
        "**Not covered by the manual**\n\n"
        "I couldn't verify an answer against the manual excerpts I retrieved, so I won't guess. "
        "Please describe the problem in a bit more detail (what you see, hear or smell, and when "
        "it happens), or have the motorcycle checked by an authorised service centre.\n\n"
        f"_(guardrail: {reason})_"
    )
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=msg)]))


def enforce_citations(callback_context: CallbackContext, llm_response: LlmResponse):
    """Reject manual-claiming answers whose citations were not actually retrieved."""
    content = llm_response.content
    if llm_response.partial or not content or not content.parts:
        return None
    if any(p.function_call for p in content.parts):
        return None  # intermediate tool-calling step
    original = "".join(p.text or "" for p in content.parts if not p.thought)
    if not original.strip():
        return None
    text = status_first(original)

    cited = _cited_labels(text)
    selected = callback_context.state.get(SELECTED)
    allowed = set(callback_context.state.get(CITABLE, {}).get(selected, []))
    unknown = sorted({c for c in cited if c not in allowed})
    if unknown:
        return _fallback(f"cited page(s) not retrieved: {', '.join(unknown)}")
    status = reply_status(text)
    if status == "answered" and not cited:
        return _fallback("answer had no page citations")
    if status is None and not cited and len(text.strip()) > UNCITED_REPLY_LIMIT:
        return _fallback("reply had neither a status line nor page citations")
    if text != original:
        return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))
    return None


async def index_uploaded_pdfs(callback_context: CallbackContext, llm_request: LlmRequest):
    """Index any PDF manual attached in chat, then replace it with a short text note.

    The PDF itself is never sent to the model; the agent reads it through search_manual like
    any other manual.
    """
    uploads = dict(callback_context.state.get(UPLOADS, {}))
    failed = dict(callback_context.state.get(FAILED_UPLOADS, {}))
    for content in llm_request.contents:
        for i, part in enumerate(content.parts or []):
            blob = part.inline_data
            if not blob or blob.mime_type != "application/pdf" or not blob.data:
                continue
            sha = hashlib.sha256(blob.data).hexdigest()
            if sha in failed:  # don't retry a failed file on every model call
                note = f"[Attached PDF could not be indexed: {failed[sha]}]"
                content.parts[i] = types.Part(text=note)
                continue
            is_new = sha not in uploads
            manual_id = uploads.get(sha) or f"{UPLOAD_PREFIX}{sha[:10]}"
            if manual_id not in retrieval.list_manual_ids():
                config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
                path = config.UPLOAD_DIR / f"{manual_id}.pdf"
                path.write_bytes(blob.data)
                name = (blob.display_name or "Uploaded manual").rsplit(".", 1)[0]
                try:
                    await asyncio.to_thread(ingest, path, name, manual_id)
                except Exception as exc:  # scanned PDF (ValueError), unreadable file, outage
                    if isinstance(exc, ValueError):
                        reason = str(exc)
                    else:
                        log.exception("indexing an attached PDF failed")
                        reason = "the file could not be read"
                    failed[sha] = reason
                    callback_context.state[FAILED_UPLOADS] = dict(failed)
                    note = f"[Attached PDF could not be indexed: {reason}]"
                    content.parts[i] = types.Part(text=note)
                    continue
            name = retrieval.load_manual(manual_id).meta["name"]
            if is_new:  # first time this session sees the file: make it the active manual
                uploads[sha] = manual_id
                callback_context.state[UPLOADS] = dict(uploads)
                callback_context.state[SELECTED] = manual_id
                note = (
                    f"[The rider attached the PDF manual '{name}'. It is indexed as manual_id "
                    f"'{manual_id}' and is now the selected manual: answer from it with "
                    "search_manual.]"
                )
            else:
                note = f"[Earlier attachment: PDF manual '{name}' (manual_id '{manual_id}').]"
            content.parts[i] = types.Part(text=note)
    return None


def build_instruction(ctx) -> str:
    names = _manual_names(ctx.state)
    listing = "\n".join(f"- {k}: {v}" for k, v in names.items()) or "- (none loaded yet)"
    selected = _current_manual(ctx.state)
    return INSTRUCTION.format(
        manuals=listing,
        selected=f"{selected} ({names[selected]})" if selected else "none",
    )


root_agent = LlmAgent(
    name="bike_manual_troubleshooter",
    # Retry transient network / 429 / 5xx failures quickly instead of failing the turn.
    model=Gemini(
        model=config.CHAT_MODEL,
        retry_options=types.HttpRetryOptions(attempts=4, initial_delay=1.0, max_delay=8.0),
    ),
    description="Answers motorcycle troubleshooting questions strictly from the bike's manual.",
    instruction=build_instruction,
    tools=[list_manuals, select_manual, search_manual],
    before_model_callback=index_uploaded_pdfs,
    after_model_callback=enforce_citations,
    generate_content_config=types.GenerateContentConfig(
        temperature=0.1, labels=config.BILLING_LABELS
    ),
)
