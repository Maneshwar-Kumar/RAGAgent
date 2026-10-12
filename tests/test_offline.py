"""Provider-free tests: chunking, keyword search, fusion and the citation guardrail.

Run with:  python -m pytest -q
"""

from types import SimpleNamespace

import numpy as np
import pytest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from bike_manual_agent import agent, retrieval
from bike_manual_agent.ingest import chunk_pages

PAGES = [
    "Acme Scooter\n12\nTROUBLESHOOTING\nEngine does not start\nIf fuel level is low, top up the fuel.",
    "Acme Scooter\n13\nMINOR MAINTENANCE TIPS\nTyre pressure front 29 psi, rear 33 psi.",
    "",  # image-only page is skipped
]


def test_chunking_keeps_printed_page_and_section():
    chunks = chunk_pages(PAGES, size=500, overlap=50)
    assert [(c.pdf_page, c.printed_page, c.section) for c in chunks] == [
        (1, "12", "TROUBLESHOOTING"),
        (2, "13", "MINOR MAINTENANCE TIPS"),
    ]


def _manual():
    chunks = [vars(c) for c in chunk_pages(PAGES, 500, 50)]
    vectors = np.eye(len(chunks), dtype=np.float32)
    bm25 = retrieval.BM25([retrieval.tokenize(c["text"]) for c in chunks])
    return retrieval.Manual("acme", {"name": "Acme"}, chunks, vectors, bm25)


def test_hybrid_search_ranks_keyword_match_first_and_labels_citations():
    manual = _manual()
    # Semantic vector points at the wrong page; the keyword ranking must still surface page 13.
    result = retrieval.search(manual, "tyre pressure", top_k=2, query_vector=np.array([1.0, 0.0]))
    assert {r["cite_as"] for r in result["results"]} == {"[p. 12]", "[p. 13]"}
    assert retrieval.BM25([["tyre", "pressure"], ["fuel"]]).scores(["tyre"])[0] > 0


def _reply(text):
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))


def _ctx(labels, manual="acme"):
    return SimpleNamespace(state={agent.SELECTED: manual, agent.CITABLE: {"acme": labels}})


@pytest.mark.parametrize(
    "text,labels,blocked",
    [
        ("**From the manual**\nTop up the fuel [p. 12].", ["p.12"], False),
        ("**From the manual**\nCheck pressure [p. 12, 13].", ["p.12", "p.13"], False),
        ("**From the manual**\nCheck pressure [p. 12, 75].", ["p.12"], True),  # 75 not retrieved
        ("**From the manual**\nCheck pressure [p. 12 and p. 13].", ["p.12", "p.13"], False),
        ("**From the manual**\nTorque to 40 Nm [p. 99].", ["p.12"], True),  # page never retrieved
        ("**From the manual**\nUsually a head gasket.", ["p.12"], True),  # no citation
        ("**Not covered by the manual**\nThe manual does not describe this.", [], False),
        ("**Need more information**\nWhich bike do you ride?", [], False),
        ("**From the manual**\nSee both pages [p. 12-13].", ["p.12", "p.13"], False),  # range
        ("**From the manual**\nSee [p. 12-14].", ["p.12", "p.13"], True),  # p.14 not retrieved
        ("From the manual:\nUsually a head gasket.", ["p.12"], True),  # status variant, no cite
        ("Usually a head gasket. " * 30, ["p.12"], True),  # long, no status line, no citation
        ("Which bike do you ride?", [], False),  # short clarification without status line
    ],
)
def test_citation_guardrail(text, labels, blocked):
    out = agent.enforce_citations(_ctx(labels), _reply(text))
    assert (out is not None) == blocked
    if blocked:
        assert out.content.parts[0].text.startswith("**Not covered by the manual**")


def test_citations_must_come_from_the_selected_manual():
    reply = _reply("**From the manual**\nTop up the fuel [p. 12].")
    assert agent.enforce_citations(_ctx(["p.12"], manual="acme"), reply) is None
    assert agent.enforce_citations(_ctx(["p.12"], manual="other-bike"), reply) is not None


def test_status_line_is_moved_to_the_top():
    text = "The photo shows white smoke.\n\n**Not covered by the manual**\nNot described [p. 12]."
    out = agent.enforce_citations(_ctx(["p.12"]), _reply(text))
    repaired = out.content.parts[0].text
    assert repaired.startswith("**Not covered by the manual**")
    assert "The photo shows white smoke." in repaired and "(guardrail:" not in repaired
    assert agent.reply_status(repaired) == "not_covered"


def test_uploaded_manuals_are_private_to_their_session(monkeypatch):
    monkeypatch.setattr(
        retrieval, "list_manual_ids", lambda index_dir=None: ["re-450", "upload-aaa", "upload-bbb"]
    )
    mine = {agent.UPLOADS: {"sha-a": "upload-aaa"}}
    assert agent.visible_manual_ids(mine) == ["re-450", "upload-aaa"]
    assert agent.visible_manual_ids({}) == ["re-450"]


def test_uploaded_pdf_is_indexed_selected_and_hidden_from_the_model(tmp_path, monkeypatch):
    import asyncio
    import json

    from google.adk.models.llm_request import LlmRequest

    from bike_manual_agent import config

    monkeypatch.setattr(config, "INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "uploads")

    def fake_ingest(path, name, manual_id):
        out = config.INDEX_DIR / manual_id
        out.mkdir(parents=True)
        (out / "meta.json").write_text(json.dumps({"manual_id": manual_id, "name": name}))
        (out / "chunks.json").write_text("[]")
        np.save(out / "embeddings.npy", np.zeros((0, 2), dtype=np.float32))

    monkeypatch.setattr(agent, "ingest", fake_ingest)
    pdf = types.Part(
        inline_data=types.Blob(mime_type="application/pdf", data=b"%PDF-1.4 x", display_name="Acme.pdf")
    )

    def request():
        return LlmRequest(contents=[types.Content(role="user", parts=[pdf, types.Part(text="hi")])])

    ctx = SimpleNamespace(state={})
    req = request()
    asyncio.run(agent.index_uploaded_pdfs(ctx, req))
    manual_id = ctx.state[agent.SELECTED]
    assert manual_id.startswith(agent.UPLOAD_PREFIX)
    assert req.contents[0].parts[0].inline_data is None  # raw PDF never reaches the model
    assert "now the selected manual" in req.contents[0].parts[0].text

    # Later turns resend the history: no re-selection if the rider switched manuals meanwhile.
    ctx.state[agent.SELECTED] = "something-else"
    req = request()
    asyncio.run(agent.index_uploaded_pdfs(ctx, req))
    assert ctx.state[agent.SELECTED] == "something-else"
    assert req.contents[0].parts[0].text.startswith("[Earlier attachment")
    assert agent.visible_manual_ids(ctx.state) == [manual_id]
