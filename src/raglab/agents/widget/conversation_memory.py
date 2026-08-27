"""What the widget remembers: one SQLite checkpointer, the thread reader behind
/api/widget/history, and the tool that recalls another experiment's conversation.

`databases/widget.db` holds internal short-term checkpoints, readable turn
logs, and selective long-term memory. It holds no
metric, decides nothing, and no board, ranking or run file may ever read it —
the widget sits outside the measured seam, and this module is why that stays
true now that the helper writes something durable.

One thread per experiment: `thread_id` is the id of the experiment the lab has
open, or `general` when it has none. That is the whole of the recall — open an
experiment and last week's conversation about it is simply there, because
SQLite kept it. Nothing is embedded or ranked.

A thread this file cannot find reads as empty, never as invented turns: the
same rule the rest of the lab keeps about a row never lying about what produced
it, applied to the one thing here that outlives a process.
"""
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock

from langchain.agents import AgentState
from langchain_core.tools import tool
from pydantic import ConfigDict, StrictBool, StrictStr
from pydantic import BaseModel

from raglab.configuration.env_settings import ROOT

#: The thread a reader is in when the lab has no experiment open.
GENERAL = 'general'

_SAVER = None
_SAVER_LOCK = RLock()

MAX_RELEVANCE_TEXT = 500


class MemoryPolicy(BaseModel):
    """The explicit, structured decision about long-term widget memory.

    The answer text is intentionally absent: saving is allowed only from this
    decision, never from a heuristic over whatever the answer happened to say.
    Strict fields and forbidden extras make a malformed model response fail
    closed at the seam that receives it.
    """

    model_config = ConfigDict(extra='forbid', strict=True)

    relevant: StrictBool = False
    should_save: StrictBool = False
    dataset_id: StrictStr = ''
    subtopic: StrictStr = ''
    reason: StrictStr = ''


def memory_policy_defaults(experiment_id: str = '') -> dict:
    """Safe state values used before a policy has been evaluated."""
    return {
        'relevant': False,
        'should_save': False,
        'dataset_id': '',
        'subtopic': '',
        'reason': '',
        'experiment_id': (experiment_id or '').strip(),
    }


# A descriptive alias keeps the state contract discoverable to callers while
# retaining the longer name for code that wants to be explicit.
widget_state_defaults = memory_policy_defaults


def relevance_guard(text: str) -> str | None:
    """Return a short refusal for deterministic, obviously bad prompts.

    This is deliberately conservative: semantic relevance remains a model
    decision. The guard only handles inputs that need no model call to reject.
    """
    import re

    value = str(text or '').strip()
    if not value:
        return 'Please ask a question about the RAG lab.'
    if len(value) > MAX_RELEVANCE_TEXT:
        return (f'That question is too long; please keep it to '
                f'{MAX_RELEVANCE_TEXT} characters or fewer.')
    unrelated = re.compile(
        r'\b(?:weather|forecast|joke|recipe|restaurant|sports? score|'
        r'horoscope|stock price|personal advice|write my essay)\b',
        re.IGNORECASE)
    if unrelated.search(value):
        return ('I can help with the RAG lab, its experiments, and RAG '
                'techniques, but not that unrelated request.')
    return None


def policy_state(policy: MemoryPolicy, experiment_id: str = '') -> dict:
    """Convert a validated policy into the agent state's flat channels."""
    normalized = policy.model_copy(update={
        'should_save': policy.relevant and policy.should_save,
    })
    return {**normalized.model_dump(),
            'experiment_id': (experiment_id or '').strip()}


class WidgetState(AgentState):
    """The agent's state beside its messages.

    Thread identity and start time, plus the structured memory-policy
    decision, are written by `thread_stamp` and `backends.ask` as part of
    `agent.invoke`'s own input. The checkpointer persists them in the same
    write as the messages, so there is no second writer racing it and
    `/api/widget/history` can report the state that produced the thread."""
    experiment_id: str   # '' in the general thread
    started_at: str      # ISO 8601, when this thread began
    relevant: bool       # structured policy decision; false before evaluation
    should_save: bool    # explicit save permission, never inferred from text
    dataset_id: str      # validated dataset context, or '' when unavailable
    subtopic: str        # policy's bounded topic label, or '' when unavailable
    reason: str          # why the policy accepted, refused, or declined save


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
            # `setup()` switches the file to write-ahead logging, which parks
            # every write in a `-wal` sidecar until a checkpoint folds it back.
            # Nothing here ever did, so `widget.db` on disk stayed a 4 KB shell
            # while the record grew beside it, and a reader opening the file in
            # a viewer saw two empty tables. Rollback mode instead: the main
            # file is always the whole record, and a `-journal` outlives no
            # transaction. The switch needs the file to itself; a second
            # process — the CLI beside the server, a one-off check — finds it
            # busy, and that is not a crash: the record works in either mode,
            # so fold what is pending into the main file and leave the switch
            # to the next opener that has the file alone.
            try:
                _SAVER.conn.execute('PRAGMA journal_mode=DELETE')
            except sqlite3.OperationalError:
                try:
                    _SAVER.conn.execute('PRAGMA wal_checkpoint(PASSIVE)')
                except sqlite3.OperationalError:
                    pass
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
    multi-block model over OpenRouter does the latter — and `backends.ask` calls
    this same function to flatten that list before the panel shows the reply.
    The two renderings are deliberately the same call: a log is a second account
    of a conversation the reader already read, and an account that words a turn
    differently from how it arrived is a quieter kind of the same lie as an
    account that leaves it out."""
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
    is `_text`'s job, and it keeps them.

    A bot turn also carries `input_tokens`/`output_tokens` when the message it
    came from has them — `usage_metadata` rides on the stored `AIMessage` and
    survives the checkpointer round-trip untouched, so nothing is computed
    here, only forwarded. A message with no `usage_metadata` (every human
    turn, and any AI turn a backend did not account for — a CLI reply keeps
    nothing at all) carries neither key, never zeros: the same distinction
    `_accounted` in `backends.py` draws between "the bill says zero" and "no
    bill was ever produced," read back on the way out instead of in."""
    out = []
    for message in messages or []:
        kind = getattr(message, 'type', '')
        text = _text(getattr(message, 'content', ''))
        if not text.strip():
            continue
        if kind == 'human':
            out.append({'role': 'you', 'text': text})
        elif kind == 'ai' and not getattr(message, 'tool_calls', None):
            turn = {'role': 'bot', 'text': text}
            used = getattr(message, 'usage_metadata', None)
            if used:
                turn['input_tokens'] = used.get('input_tokens')
                turn['output_tokens'] = used.get('output_tokens')
            out.append(turn)
    return out


def _channels(name: str) -> dict:
    """One thread's persisted state, or an empty mapping for a thread that has
    never been used. Two readers need this — the history route and the stamp
    that decides whether a thread has already begun — and a thread nobody has
    used has to read the same way for both, so the "no checkpoint is not an
    error" rule lives here once rather than in each of them."""
    checkpoint = saver().get({'configurable': {'thread_id': name}})
    return (checkpoint or {}).get('channel_values') or {}


def thread_stamp(thread: str, now: datetime | None = None) -> dict:
    """What a turn writes about the thread it lands in, for `backends.ask` to
    pass through `agent.invoke`'s input.

    `experiment_id` is the thread itself, because that is what a thread *is*
    here — one conversation per experiment — and `''` on the general thread,
    which belongs to no experiment. It is written on every turn rather than
    only the first: it cannot change for a given thread, so re-stating it
    costs nothing and repairs a thread whose first turn predates this stamp.

    `started_at` is the opposite: it is stamped only when the thread has none,
    and is left out of the returned mapping entirely otherwise, so langgraph
    leaves that channel exactly as it found it. A "when this began" that moved
    to the latest turn would be a field naming itself after something it is
    not — worse than the empty string it replaced, because an empty string at
    least admits to knowing nothing. Two turns racing to open the same thread
    could both find it unstamped and both write; they would be writing very
    nearly the same instant, and the loser's value is overwritten rather than
    added to, so the field still names one moment near the thread's start."""
    name = (thread or '').strip() or GENERAL
    stamp = {'experiment_id': '' if name == GENERAL else name}
    if not _channels(name).get('started_at'):
        stamp['started_at'] = (now or datetime.now(timezone.utc)).isoformat(
            timespec='seconds')
    return stamp


def history(thread: str) -> dict:
    """One thread, as the model holds it. A thread nobody has used reads as
    empty rather than as an error: a conversation that has not happened yet is
    not a failure, and the empty log with its starters says so. Each bot turn
    carries its token account when `_turns` found one on the message it came
    from — this is a pass-through, not a second computation, so the account a
    reader sees on a redraw is the very same one the live reply reported."""
    name = (thread or '').strip() or GENERAL
    values = _channels(name)
    return {'thread': name,
            'experiment_id': values.get('experiment_id') or '',
            'started_at': values.get('started_at') or '',
            'turns': _turns(values.get('messages'))}


def trace(thread: str) -> dict:
    """One thread as the *model* had it, step by step: the system lines it was
    handed, the reader's question, each tool call with its arguments, what the
    tool said back, and the reply — where `history` renders only the two turns
    the reader saw. For the developer page; a reader is never shown this."""
    name = (thread or '').strip() or GENERAL
    values = _channels(name)
    steps = []
    for message in values.get('messages') or []:
        kind = {'system': 'system', 'human': 'human', 'ai': 'ai',
                'tool': 'tool'}.get(getattr(message, 'type', ''), 'other')
        step = {'kind': kind, 'text': _text(getattr(message, 'content', ''))}
        if kind == 'ai':
            calls = getattr(message, 'tool_calls', None) or []
            if calls:
                step['tool_calls'] = [{'name': c.get('name'),
                                       'args': c.get('args'),
                                       'id': c.get('id')} for c in calls]
            used = getattr(message, 'usage_metadata', None) or {}
            if used:
                step['input_tokens'] = used.get('input_tokens')
                step['output_tokens'] = used.get('output_tokens')
        elif kind == 'tool':
            step['name'] = getattr(message, 'name', None)
            step['tool_call_id'] = getattr(message, 'tool_call_id', None)
        steps.append(step)
    return {'thread': name,
            'experiment_id': values.get('experiment_id') or '',
            'started_at': values.get('started_at') or '',
            'dataset_id': values.get('dataset_id') or '',
            'steps': steps}


def threads() -> list[str]:
    """Every thread the log holds, newest write first. Read straight from the
    checkpointer's own table: it is the one list of conversations there is."""
    rows = saver().conn.execute(
        'SELECT thread_id, MAX(checkpoint_id) AS latest FROM checkpoints '
        'GROUP BY thread_id ORDER BY latest DESC').fetchall()
    return [row[0] for row in rows]


def forget(thread: str) -> None:
    """Delete one thread. New Chat's whole implementation: the conversation you
    are in ends, and every other experiment's is untouched."""
    name = (thread or '').strip() or GENERAL
    saver().delete_thread(name)


#: How many turns one recall hands the model. The same reasoning as
#: `experiment_tools.MAX_LISTED`: a call that could return every turn of a long
#: conversation would spend the context window on its tail.
MAX_RECALLED = 20


@tool
def recall_conversation(experiment_id: str) -> str:
    """What was said about one experiment before; the model-facing prompt is
    fixtures/prompts/widget_tools.yaml's entry."""
    name = (experiment_id or '').strip()
    if not name:
        return ('Name the experiment to recall. The conversation you are in '
                'now needs no recalling — it is already the context.')
    turns = history(name)['turns']
    if not turns:
        return (f'There is no recorded conversation about {name}. That is not '
                'the same as nothing being true of it: the experiment may well '
                'exist — read_experiment answers that — it has simply never '
                'been discussed here.')
    shown, dropped = turns[-MAX_RECALLED:], max(0, len(turns) - MAX_RECALLED)
    lines = [f'The conversation about {name}, '
             + (f'its last {len(shown)} turns; {dropped} earlier turn(s) are '
                f'not shown, because one recall is capped at {MAX_RECALLED}.'
                if dropped else f'all {len(shown)} turns of it.')]
    # Flipped on purpose from `_turns`'s own roles ('you' for the human, 'bot'
    # for the model): this text is handed to the model about a conversation it
    # was not just in, so "you" here has to mean the model, and the reader who
    # asked the human turns belongs to becomes the third party, "reader". Do
    # not "fix" this to match `_turns` — that would make the model refer to
    # its own past words as someone else's.
    for turn in shown:
        lines.append(('reader: ' if turn['role'] == 'you' else 'you said: ')
                     + turn['text'])
    return '\n'.join(lines)
