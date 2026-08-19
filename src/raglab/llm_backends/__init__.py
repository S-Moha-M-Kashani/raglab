"""Helper for ALL model calls, every stage: the one chat-model factory, the
CLI subprocess drive it dispatches to, the role catalogue saying which model
runs which stage, and the panel-typed OpenRouter key (process memory only)."""
from . import (chat_model_factory, cli_subprocess_chat,  # noqa: F401
               model_role_catalogue, openrouter_key_memory)
