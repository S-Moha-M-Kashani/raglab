"""The lab's third kind of backend: a CLI on this machine (`claude`/`codex`), run as a subprocess in place of an HTTP endpoint, reaching a strong model with no API key at the price of a process spawn per call.
Driven as a completion endpoint rather than as an agent — every flag near its call site closes one specific way a row could otherwise be wrong; see `_KEEP`/`_child_env` and each `subprocess` call below.
Nothing here falls back: a non-zero exit, a timeout, an error envelope or an empty reply all raise `CliError` rather than being read as "no opinion".
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages import convert_to_openai_messages
from langchain_core.outputs import ChatGeneration, ChatResult

# Appended to whatever system message the stage already wrote: an agent CLI's
# default habit is to explain itself, and every stage here parses with a regex.
API_LINE = ('You are being called as a text completion endpoint, not an '
            'assistant. Follow the instruction exactly and emit only the '
            'requested output: no preamble, no explanation, no code fences, '
            'no tool use, no questions.')


# The only variables a call is allowed to see. An **allowlist**, not a denylist:
# both CLIs read their configuration from the environment, and a variable like
# `ANTHROPIC_BASE_URL` or `ANTHROPIC_DEFAULT_SONNET_MODEL` can redirect the call
# or remap the model the row is labelled with. A denylist would have to be
# re-read every time either CLI ships a new variable, and fails silently when
# it isn't — the one failure this backend cannot have. Each entry here earns
# its place by being needed to *reach* the model, and none of them can change
# which model answers.
_KEEP = (
    'PATH',                 # find the command, and whatever it execs
    'HOME',                 # where the login lives: ~/.claude, ~/.codex
    'CLAUDE_CONFIG_DIR',    # …unless it was moved. Credential location, not
    'CODEX_HOME',           #    configuration: --ignore-user-config still holds
    'TMPDIR',               # the per-user temp dir on macOS
    'USER', 'LOGNAME', 'SHELL',   # identity, which both CLIs read for their logs
    'LANG', 'LC_ALL', 'LC_CTYPE',  # the child's own idea of text (see `encoding`)
)
_KEEP_PREFIX = ('XDG_',)    # where a Linux box keeps that same login

# Deliberately *not* kept: HTTPS_PROXY, NODE_EXTRA_CA_CERTS and their kin. A
# machine that needs one gets a call that cannot connect and raises CliError —
# the opposite failure of a silently redirected base url.


def _child_env() -> dict[str, str]:
    """The environment one call runs with: `_KEEP`, and nothing else."""
    return {name: value for name, value in os.environ.items()
            if name in _KEEP or name.startswith(_KEEP_PREFIX)}


class CliError(RuntimeError):
    """A CLI backend did not produce a reply this lab can use — one class for every failure, with a reason."""


def _claude_argv(model: str, effort: str, instructions: str) -> list[str]:
    return ['claude', '-p', '--output-format', 'json',
            '--model', model, '--effort', effort,
            '--tools', '',                      # a judge must not read .runs/
            '--no-session-persistence',         # the lab accounts for its own files
            '--setting-sources', '',            # no CLAUDE.md, hooks or skills
            '--strict-mcp-config', '--disable-slash-commands',
            '--system-prompt', instructions]    # replaces the agent prompt


def _codex_argv(model: str, effort: str, instructions: str) -> list[str]:
    # `instructions` is unused: codex has no flag that replaces its system
    # prompt, so its text goes into stdin instead (see `_codex_stdin`). The
    # signature is shared so `CLIS` stays a table of two like-shaped rows.
    return ['codex', 'exec', '--json',
            '-s', 'read-only',                  # the sandbox equivalent of --tools ''
            '--ephemeral', '--skip-git-repo-check',
            '--ignore-rules', '--ignore-user-config',
            '-m', model, '-c', f'model_reasoning_effort="{effort}"',
            '-']                                # read the prompt from stdin


def _read_claude(stdout: str) -> tuple[str, dict]:
    """The envelope's `result`, byte-exact, or a refusal saying why not.

    `stop_reason == 'max_tokens'` is refused: a grade list truncated mid-way
    would otherwise parse like a complete one, leaving the unreached entries
    read as "no opinion" and scored 0.5. Only that value is refused, not every
    non-`end_turn` one, so a stop reason a later version ships isn't mistaken
    for a truncation."""
    envelope = json.loads(stdout)
    if envelope.get('is_error'):
        raise CliError(f'claude reported an error: '
                       f'{envelope.get("result") or envelope}')
    if envelope.get('stop_reason') == 'max_tokens':
        raise CliError('claude stopped at its output limit, so the reply is a '
                       'fragment — and a fragment of a grade list scores every '
                       'document it never reached 0.5, which clears the gate')
    spent = envelope.get('usage') or {}
    return envelope.get('result') or '', spent


def _read_codex(stdout: str) -> tuple[str, dict]:
    """The last agent message and the usage the turn ended with — last, not first, or a preamble would be scored.

    Unparseable lines are skipped rather than fatal: the stream is a log, and a
    new line the lab does not know about must not end a judged run."""
    text, spent = '', {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        item = event.get('item') or {}
        if (event.get('type') == 'item.completed'
                and item.get('type') == 'agent_message'):
            text = item.get('text') or ''
        if event.get('type') == 'turn.completed':
            spent = event.get('usage') or {}
    return text, spent


@dataclass(frozen=True)
class CliSpec:
    """One CLI: what to run, what it accepts, and how to read what it says."""
    binary: str
    efforts: tuple[str, ...]
    argv: Callable[[str, str, str], list[str]]   # model, effort, instructions
    stdin: Callable[[str, str], str]             # instructions, body
    read: Callable[[str], tuple[str, dict]]      # stdout -> text, usage


CLIS: dict[str, CliSpec] = {
    'claude': CliSpec(
        binary='claude',
        efforts=('low', 'medium', 'high', 'xhigh', 'max'),
        argv=_claude_argv,
        # The instructions went in the argv, so stdin is the question alone.
        stdin=lambda instructions, body: body,
        read=_read_claude),
    'codex': CliSpec(
        binary='codex',
        efforts=('low', 'medium', 'high', 'none'),
        argv=_codex_argv,
        stdin=lambda instructions, body: f'{instructions}\n\n{body}'.strip(),
        read=_read_codex),
}


def cli_available(cli: str) -> bool:
    """Whether this machine has the command — the whole check, since an alias cannot be verified without a call."""
    spec = CLIS.get(cli)
    return bool(spec and shutil.which(spec.binary))


def checked_effort(cli: str, effort: str) -> str:
    """The effort value, or a refusal naming what this CLI accepts.

    The two CLIs accept different values, and an unaccepted one exits 0 with no
    text — refused here rather than letting a stage read that as "no opinion"
    and score everything 0.5."""
    spec = CLIS[cli]
    if effort not in spec.efforts:
        raise ValueError(
            f'{effort!r} is not a reasoning effort {spec.binary} accepts; '
            'expected one of ' + ', '.join(repr(v) for v in spec.efforts)
            + '. Measured: an unaccepted value exits 0 and answers nothing, so '
            'this refuses rather than letting a stage score everything 0.5')
    return effort


def _instructions_and_body(messages: list[BaseMessage]) -> tuple[str, str]:
    """The system text and everything else, flattened; a non-user/system turn is labelled, not silently merged."""
    system, body = [], []
    for message in convert_to_openai_messages(messages):
        content = message.get('content') or ''
        if not isinstance(content, str):
            content = ''.join(part.get('text', '') for part in content
                              if isinstance(part, dict))
        role = message.get('role')
        if role == 'system':
            system.append(content)
        elif role == 'user':
            body.append(content)
        else:
            body.append(f'{role}: {content}')
    return '\n\n'.join(system).strip(), '\n\n'.join(body).strip()


# A whole reply that is one fenced block, and nothing more. Anchored at both
# ends on purpose — see `_unfence`.
_WHOLE_FENCE = re.compile(r'\A```[A-Za-z0-9_+-]*\n(?P<body>.*?)\n?```\Z', re.S)


def _unfence(text: str) -> str:
    """Unwrap a reply that is *entirely* one code fence — a habit of the transport, not a different measurement.

    A reply that merely contains a fence is returned byte-exact: cleaning those
    up too is how a row measured on claude stops being comparable with ollama."""
    match = _WHOLE_FENCE.match(text)
    if not match or '```' in match.group('body'):
        return text
    return match.group('body').strip()


def _run(spec: CliSpec, argv: list[str], prompt: str, timeout: int) -> tuple[str, dict]:
    """One call. Every way it can fail raises, and each says which."""
    # A fresh directory per call, not one per process, so nothing one call
    # leaves behind can be read by the next.
    with tempfile.TemporaryDirectory() as empty:
        try:
            # `encoding='utf-8'` rather than `text=True` (locale-dependent):
            # under a C/POSIX locale that is ASCII, and this corpus is Farsi, so
            # a correct answer would fail to decode — or under latin-1 it would
            # decode as silent mojibake. `errors='strict'` for the same reason
            # nothing else here falls back: a replaced byte changes the text
            # that gets scored with no field on the row saying so.
            done = subprocess.run(argv, input=prompt, capture_output=True,
                                  encoding='utf-8', errors='strict',
                                  timeout=timeout, cwd=empty, env=_child_env())
        except FileNotFoundError as error:
            raise CliError(f'{spec.binary} is not installed on this machine, so '
                           'this backend cannot run — install it, or pick a '
                           'backend that is here') from error
        except subprocess.TimeoutExpired as error:
            raise CliError(f'{spec.binary} did not answer within {timeout}s') \
                from error
        except UnicodeDecodeError as error:
            raise CliError(f'{spec.binary} wrote bytes that are not UTF-8, and '
                           'guessing an encoding would change the text that '
                           f'gets scored: {error}') from error
    if done.returncode != 0:
        detail = (done.stderr or done.stdout or '').strip()[:400]
        raise CliError(f'{spec.binary} exited {done.returncode}: {detail}')
    try:
        text, spent = spec.read(done.stdout)
    except CliError:
        raise
    except Exception as error:
        raise CliError(f'{spec.binary} wrote a reply this lab cannot read: '
                       f'{done.stdout[:200]!r}') from error
    text = _unfence(text.strip())
    if not text:
        raise CliError(
            f'{spec.binary} answered with no text at all. Measured once '
            'already: a reasoning effort it does not accept exits 0 and says '
            'nothing, and a stage reading that tolerantly scores every '
            'document 0.5 — which clears the gate threshold and produces a '
            'gated row that was measured ungated')
    return text, spent


class CliChat(BaseChatModel):
    """A chat model whose transport is a subprocess.

    Sync only: `BaseChatModel`'s default `_agenerate` runs `_generate` in a
    thread, the right shape since the thing being parallelised is a process
    (`ragas_eval.JUDGE_LOAD` caps it at three). No `bind_tools` — unlike
    `FakeChat`, this backend genuinely cannot do tool calling.
    """

    cli: str
    model: str
    effort: str = 'low'
    timeout: int = 600

    @property
    def _llm_type(self) -> str:
        return f'cli-{self.cli}'

    def _generate(self, messages: list[BaseMessage],
                  stop: list[str] | None = None,
                  run_manager: Optional[CallbackManagerForLLMRun] = None,
                  **kwargs: Any) -> ChatResult:
        spec = CLIS[self.cli]
        # The lab's convention: one client per run, a per-stage model on the
        # request (models.ROLES). For a subprocess that means the argv.
        model = kwargs.get('model') or self.model
        system, body = _instructions_and_body(messages)
        instructions = f'{system}\n\n{API_LINE}'.strip()
        text, spent = _run(spec, spec.argv(model, self.effort, instructions),
                           spec.stdin(instructions, body), self.timeout)
        spent_in = int(spent.get('input_tokens') or 0)
        spent_out = int(spent.get('output_tokens') or 0)
        message = AIMessage(content=text, usage_metadata={
            'input_tokens': spent_in, 'output_tokens': spent_out,
            'total_tokens': spent_in + spent_out})
        return ChatResult(generations=[ChatGeneration(message=message)])
