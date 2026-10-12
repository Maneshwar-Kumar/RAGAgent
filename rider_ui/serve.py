"""Start the rider UI with the agent already warm:  python -m rider_ui.serve

Before the Streamlit server starts listening, this process loads every bundled index, creates
the Vertex AI client and the ADK runner, and pre-embeds common searches. Cloud Run counts this
as startup, so the first rider of a fresh instance doesn't wait for it, and the app's sessions
share the warm objects because they run in this same process.
`streamlit run rider_ui/app.py` still works; it just warms up lazily on the first question.
"""

import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit.web import bootstrap  # noqa: E402

from rider_ui import backend  # noqa: E402

log = logging.getLogger("rider_ui.serve")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if os.environ.get("RAGAGENT_WARMUP", "1") != "0":
        backend.get_backend()
        log.info("warm-up: %s", backend.warm_up())
    flags = {
        "server_port": int(os.environ.get("PORT", "8501")),
        "server_address": os.environ.get("RAGAGENT_ADDRESS", "0.0.0.0"),
        "server_headless": True,
    }
    bootstrap.load_config_options(flag_options=flags)
    bootstrap.run(str(ROOT / "rider_ui" / "app.py"), False, [], flags)


if __name__ == "__main__":
    main()
