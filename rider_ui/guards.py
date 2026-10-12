"""Input limits and cost guards for the public demo (process-wide, thread-safe).

Cloud Run runs a single instance (max-instances 1), so in-process counters bound the
model spend of the hosted demo. All limits are environment variables.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Limits:
    question_words: int = 100
    message_characters: int = 2000
    photo_megabytes: int = 5
    photos_per_conversation: int = 3
    manual_megabytes: int = 40
    manual_pages: int = 400
    daily_questions: int = 300
    daily_uploads: int = 10
    questions_per_minute: int = 8
    turns_per_conversation: int = 30
    # Daily cap on model spend, metered from the token counts Gemini reports. Prices are
    # INR per million tokens from the Cloud Billing catalog for Gemini 3.8 Flash on the
    # global endpoint (text/image input 144, cached input 14, output incl. thinking 720).
    daily_model_budget_inr: float = 1500.0
    price_input_inr_per_mtok: float = 144.0
    price_cached_inr_per_mtok: float = 14.0
    price_output_inr_per_mtok: float = 720.0

    @classmethod
    def from_env(cls) -> "Limits":
        return cls(
            daily_questions=_env_int("RAGAGENT_DAILY_QUESTION_CAP", cls.daily_questions),
            daily_uploads=_env_int("RAGAGENT_DAILY_UPLOAD_CAP", cls.daily_uploads),
            questions_per_minute=_env_int("RAGAGENT_RATE_PER_MIN", cls.questions_per_minute),
            turns_per_conversation=_env_int("RAGAGENT_MAX_TURNS", cls.turns_per_conversation),
            daily_model_budget_inr=_env_float(
                "RAGAGENT_DAILY_MODEL_BUDGET_INR", cls.daily_model_budget_inr
            ),
            price_input_inr_per_mtok=_env_float(
                "RAGAGENT_PRICE_INPUT_INR_PER_MTOK", cls.price_input_inr_per_mtok
            ),
            price_cached_inr_per_mtok=_env_float(
                "RAGAGENT_PRICE_CACHED_INR_PER_MTOK", cls.price_cached_inr_per_mtok
            ),
            price_output_inr_per_mtok=_env_float(
                "RAGAGENT_PRICE_OUTPUT_INR_PER_MTOK", cls.price_output_inr_per_mtok
            ),
        )


LIMITS = Limits.from_env()


def cost_inr(usage: dict, limits: Limits = LIMITS) -> float:
    """Model cost of one answer from its token counts (thinking is billed as output)."""
    cached = usage.get("cached", 0)
    fresh = max(usage.get("prompt", 0) - cached, 0)
    output = usage.get("output", 0) + usage.get("thinking", 0)
    return (
        fresh * limits.price_input_inr_per_mtok
        + cached * limits.price_cached_inr_per_mtok
        + output * limits.price_output_inr_per_mtok
    ) / 1_000_000


def check_question(text: str, limits: Limits = LIMITS) -> str | None:
    """Return a rider-facing reason the question cannot be sent, or None."""
    words = len(text.split())
    if not words:
        return "Type a question first."
    if len(text) > limits.message_characters:
        return f"Please keep your message under {limits.message_characters:,} characters."
    if words > limits.question_words:
        return f"Please shorten your question to {limits.question_words} words or fewer ({words} now)."
    return None


class Guard:
    """Per-visitor rate limiting plus global daily caps (UTC day)."""

    def __init__(self, limits: Limits = LIMITS, clock=time.time):
        self.limits = limits
        self._clock = clock
        self._lock = threading.Lock()
        self._recent: dict[str, deque] = {}
        self._day = None
        self._questions = 0
        self._uploads = 0
        self._spent_inr = 0.0

    def _roll_day(self) -> None:
        day = datetime.fromtimestamp(self._clock(), timezone.utc).date()
        if day != self._day:
            self._day, self._questions, self._uploads, self._spent_inr = day, 0, 0, 0.0

    @property
    def spent_today_inr(self) -> float:
        with self._lock:
            self._roll_day()
            return self._spent_inr

    def record_usage(self, usage: dict) -> float:
        """Add one answer's model cost to today's spend; returns that cost in INR."""
        cost = cost_inr(usage, self.limits)
        with self._lock:
            self._roll_day()
            self._spent_inr += cost
        return cost

    def admit_question(self, visitor: str) -> str | None:
        """Record a question if allowed; otherwise return the reason it is refused."""
        with self._lock:
            self._roll_day()
            if self._questions >= self.limits.daily_questions:
                return "The demo has reached today's question limit. Please try again tomorrow."
            if self._spent_inr >= self.limits.daily_model_budget_inr:
                return "The demo has reached today's usage limit. Please try again tomorrow."
            now = self._clock()
            window = self._recent.setdefault(visitor, deque())
            while window and now - window[0] > 60:
                window.popleft()
            if len(window) >= self.limits.questions_per_minute:
                return "You're sending questions quickly. Please wait a minute and try again."
            window.append(now)
            self._questions += 1
            if len(self._recent) > 5000:  # forget idle visitors
                for key in [k for k, v in self._recent.items() if not v or now - v[-1] > 60]:
                    del self._recent[key]
            return None

    def admit_upload(self) -> str | None:
        with self._lock:
            self._roll_day()
            if self._uploads >= self.limits.daily_uploads:
                return "The demo has reached today's manual-upload limit. Please try again tomorrow."
            self._uploads += 1
            return None
