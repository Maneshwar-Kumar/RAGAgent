from . import config

config.use_access_token_if_provided()

from . import agent  # noqa: E402,F401  (ADK discovers `agent.root_agent`)
