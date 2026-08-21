"""What the widget remembers: one SQLite checkpointer, the thread reader behind
/api/widget/history, and the tool that recalls another experiment's conversation.

`databases/widget.db` is a conversation log and nothing else. It holds no
metric, decides nothing, and no board, ranking or run file may ever read it —
the widget sits outside the measured seam, and this module is why that stays
true now that the helper writes something durable.

One thread per experiment: `thread_id` is the id of the experiment the lab has
open, or `general` when it has none. That is the whole of the recall — open an
experiment and last week's conversation about it is simply there, because
SQLite kept it. Nothing is embedded, ranked or summarised.

A thread this file cannot find reads as empty, never as invented turns: the
same rule the rest of the lab keeps about a row never lying about what produced
it, applied to the one thing here that outlives a process.
"""
import os
import sqlite3
from pathlib import Path
from threading import RLock

from langchain.agents import AgentState

from raglab.configuration.env_settings import ROOT

#: The thread a reader is in when the lab has no experiment open.
GENERAL = 'general'

_SAVER = None
_SAVER_LOCK = RLock()


class WidgetState(AgentState):
    """The agent's state beside its messages. Deliberately two fields: the
    state is a real thing that persists and can be read back, and small enough
    that redesigning it later is a rewrite of this class rather than an
    unpicking of everything that grew into it."""
    experiment_id: str   # '' in the general thread
    started_at: str      # ISO 8601, when this thread began


def db_path(env: dict | None = None) -> Path:
    """Where conversations live. `RAGLAB_WIDGET_DB` overrides, which is what
    lets the suite point every test at a temp file from one autouse fixture."""
    environ = os.environ if env is None else env
    override = (environ.get('RAGLAB_WIDGET_DB') or '').strip()
    return Path(override) if override else ROOT / 'databases' / 'widget.db'


def saver():
    """The process's one checkpointer, built on first use rather than at import
    so the suite can redirect the path before anything opens a file."""
    global _SAVER
    with _SAVER_LOCK:
        if _SAVER is None:
            from langgraph.checkpoint.sqlite import SqliteSaver
            target = db_path()
            target.parent.mkdir(parents=True, exist_ok=True)
            # check_same_thread=False because the panel answers on a threadpool:
            # this connection belongs to the process, not to one request.
            _SAVER = SqliteSaver(sqlite3.connect(str(target),
                                                 check_same_thread=False))
            _SAVER.setup()
        return _SAVER


def close() -> None:
    """Drop the checkpointer so the next use reopens the file. For tests, and
    for the one thing a test cannot otherwise stage: a restart of the lab."""
    global _SAVER
    with _SAVER_LOCK:
        if _SAVER is not None:
            _SAVER.conn.close()
        _SAVER = None


def _text(content) -> str:
    """One message rendered the way the reader already saw it live. A model may
    answer with a plain string or with a list of content blocks — a reasoning or
    multi-block model over OpenRouter does the latter — and `backends.ask`
    flattens that list to exactly this joined text before the panel shows it.
    The two renderings are deliberately the same one: a log is a second account
    of a conversation the reader already read, and an account that words a turn
    differently from how it arrived is a quieter kind of the same lie as an
    account that leaves it out. (When the live path is rewired onto this module,
    it should call this rather than keep its own copy of the join.)"""
    if isinstance(content, list):
        return ' '.join(part.get('text', '') if isinstance(part, dict)
                        else str(part) for part in content)
    return content if isinstance(content, str) else str(content)


def _turns(messages) -> list[dict]:
    """The conversation as a reader saw it: what was asked, what was answered.

    Two kinds of message are left out, for two unrelated reasons. An assistant
    message carrying tool calls is dropped because a tool call is how an answer
    was reached rather than part of the answer — the reader was never shown it,
    and a log is what the reader saw. A message with no text at all is dropped
    because there is nothing of it to show; that is what an assistant message
    looks like in the moment it calls a tool instead of speaking.

    Neither rule may swallow a reply that simply arrives in another shape. This
    once tested `isinstance(text, str)` and dropped everything else, which meant
    a block-shaped answer the reader had watched arrive was gone from the log by
    the next day, leaving a question with nothing under it — a record that
    misrepresents the conversation as surely as an invented turn would. Rendering
    is `_text`'s job, and it keeps them."""
    out = []
    for message in messages or []:
        kind = getattr(message, 'type', '')
        text = _text(getattr(message, 'content', ''))
        if not text.strip():
            continue
        if kind == 'human':
            out.append({'role': 'you', 'text': text})
        elif kind == 'ai' and not getattr(message, 'tool_calls', None):
            out.append({'role': 'bot', 'text': text})
    return out


def history(thread: str) -> dict:
    """One thread, as the model holds it. A thread nobody has used reads as
    empty rather than as an error: a conversation that has not happened yet is
    not a failure, and the empty log with its starters says so."""
    name = (thread or '').strip() or GENERAL
    checkpoint = saver().get({'configurable': {'thread_id': name}})
    values = (checkpoint or {}).get('channel_values') or {}
    return {'thread': name,
            'experiment_id': values.get('experiment_id') or '',
            'started_at': values.get('started_at') or '',
            'turns': _turns(values.get('messages'))}


def forget(thread: str) -> None:
    """Delete one thread. New Chat's whole implementation: the conversation you
    are in ends, and every other experiment's is untouched."""
    name = (thread or '').strip() or GENERAL
    saver().delete_thread(name)
