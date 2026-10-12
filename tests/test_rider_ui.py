"""Provider-free tests for the MotoMedic rider UI (Streamlit AppTest + a fake backend)."""

from concurrent.futures import Future
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from rider_ui import backend
from rider_ui.guards import Guard, Limits, check_question

APP = str(Path(__file__).resolve().parents[1] / "rider_ui" / "app.py")

HIMALAYAN = backend.Manual(
    manual_id="royal-enfield-himalayan-450",
    name="Royal Enfield Himalayan 450 (BS6) Owner's Manual",
    model_label="Himalayan 450 BS6",
    short_name="Himalayan",
    edition="RAM00818/A, 4 November 2025, 149 pages",
    edition_short="Owner's manual · 2025 edition",
    brand="Royal Enfield",
    official_site="https://www.royalenfield.com",
    source_url="https://www.royalenfield.com/manual.pdf",
    select_label="Royal Enfield Himalayan 450",
    pill_label="Himalayan BS6",
    confirm_label="My motorcycle is a Himalayan BS6",
    source_title="Owner’s manual · English",
    edition_label="RAM00818/A · 4 November 2025",
    pages=149,
    chunks=179,
)

ANSWER = backend.Answer(
    status="answered",
    body="Check the ignition / engine kill switch is ON [p. 119].",
    citations=(backend.Citation("p. 119", 121, "TROUBLESHOOTING", "Engine does not start ..."),),
    searches=("engine does not start",),
    elapsed_s=4.2,
    reference="e-123",
)


class FakeBackend:
    def __init__(self, answer=ANSWER):
        self.answer = answer
        self.questions = []

    def manuals(self):
        return [HIMALAYAN]

    def start_conversation(self, user_id, manual):
        return "session-1"

    def ask(self, user_id, session_id, text, photo, progress):
        self.questions.append((text, photo))
        future = Future()
        future.set_result(self.answer)
        return future


@pytest.fixture
def fake(monkeypatch):
    fake_backend = FakeBackend()
    monkeypatch.setattr(backend, "get_backend", lambda: fake_backend)
    return fake_backend


def test_layout_matches_motomedic_and_input_waits_for_confirmation(fake):
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value == "MotoMedic handles your routine queries for your Himalayan"
    assert at.selectbox[0].label == "Motorcycle"
    assert at.checkbox[0].label == "My motorcycle is a Himalayan BS6"
    assert any("What would you like to know?" in m.value for m in at.markdown)
    assert at.chat_input[0].disabled  # scope confirmation required first
    at.checkbox[0].check().run()
    assert not at.chat_input[0].disabled


def test_question_renders_cited_answer(fake):
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.checkbox[0].check().run()
    at.chat_input[0].set_value("My bike won't start").run()
    assert not at.exception
    assert fake.questions == [("My bike won't start", None)]
    assert [c.value for c in at.caption if c.value == "Answer"] == ["Answer"]
    assert any("[p. 119]" in m.value for m in at.markdown)
    assert any(e.label == "Source · p. 119" for e in at.expander)


def test_sample_photo_is_sent_with_the_white_smoke_question(fake):
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.checkbox[0].check().run()
    next(b for b in at.button if "sample photo" in b.label).click().run()
    text, photo = fake.questions[0]
    assert text == "Why is this smoke coming from my bike?"
    assert photo[1] == "image/jpeg" and photo[0][:3] == b"\xff\xd8\xff"


def test_long_question_is_refused_before_reaching_the_agent(fake):
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.checkbox[0].check().run()
    at.chat_input[0].set_value("word " * 101).run()
    assert fake.questions == []
    assert any("100 words" in w.value for w in at.warning)


def test_parse_answer_statuses():
    assert backend.parse_answer("**From the manual**\nDo X [p. 1].") == ("answered", "Do X [p. 1].")
    assert backend.parse_answer("**Not covered by the manual**\nNo.")[0] == "not_covered"
    assert backend.parse_answer("**Need more information**\nWhich bike?")[0] == "need_info"
    assert backend.parse_answer("**Not covered by the manual**\n...(guardrail: x)")[0] == "guardrail"


def test_guard_limits():
    assert check_question("   ") is not None
    assert check_question("short question") is None
    now = [0.0]
    guard = Guard(Limits(questions_per_minute=2, daily_questions=3), clock=lambda: now[0])
    assert guard.admit_question("a") is None
    assert guard.admit_question("a") is None
    assert "wait a minute" in guard.admit_question("a")
    now[0] = 61.0
    assert guard.admit_question("a") is None
    assert "today's question limit" in guard.admit_question("b")
