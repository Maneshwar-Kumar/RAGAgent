"""Runtime configuration, read from environment variables (see env.sample)."""

import os
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Where ingested manuals are stored. One sub-folder per manual.
INDEX_DIR = Path(os.getenv("RAGAGENT_INDEX_DIR", REPO_ROOT / "index"))
UPLOAD_DIR = Path(os.getenv("RAGAGENT_UPLOAD_DIR", REPO_ROOT / "manuals" / "uploads"))

# Models (Vertex AI by default). With a Gemini API key, the chat model name works as is, but
# text-embedding-005 is Vertex-only: set RAGAGENT_EMBEDDING_MODEL and re-index the manuals.
CHAT_MODEL = os.getenv("RAGAGENT_CHAT_MODEL", "gemini-3.8-flash")
EMBEDDING_MODEL = os.getenv("RAGAGENT_EMBEDDING_MODEL", "text-embedding-005")

# Retrieval knobs.
CHUNK_CHARS = int(os.getenv("RAGAGENT_CHUNK_CHARS", "1200"))
CHUNK_OVERLAP = int(os.getenv("RAGAGENT_CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("RAGAGENT_TOP_K", "6"))

USE_VERTEX = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").strip().lower() in {"1", "true", "yes"}
# Vertex AI request labels show up in billing reports (not supported with a Gemini API key).
BILLING_LABELS = {"app": os.getenv("RAGAGENT_BILLING_LABEL", "ragagent")} if USE_VERTEX else None


def use_access_token_if_provided() -> None:
    """Optionally authenticate Vertex AI with a short-lived OAuth access token.

    When GOOGLE_OAUTH_ACCESS_TOKEN is set (for example to the output of
    `gcloud auth print-access-token`), Application Default Credentials are
    replaced by that token for this process. Without it, normal ADC is used.
    """
    token = os.getenv("GOOGLE_OAUTH_ACCESS_TOKEN")
    if not token:
        return
    import google.auth
    from google.oauth2.credentials import Credentials

    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    google.auth.default = lambda *args, **kwargs: (Credentials(token), project)


@lru_cache(maxsize=1)
def genai_client():
    """Return a google-genai client configured from the environment.

    Uses Vertex AI when GOOGLE_GENAI_USE_VERTEXAI is true (project/location from
    GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION and Application Default
    Credentials), otherwise a Gemini API key from GOOGLE_API_KEY.
    """
    from google import genai

    return genai.Client()
