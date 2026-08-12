"""The lab's third kind of backend: a CLI on this machine, run as a subprocess.

`openrouter` and `ollama` are both OpenAI-compatible HTTP endpoints, which is
why `llm._endpoint` can hold the whole difference between them. `claude` and
`codex` are neither. They are agent CLIs that this machine already has and is
already logged into, so the lab can reach a strong model with **no API key and
no new dependency** — which matters because the four deciding metrics are
judged, and an unkeyed lab measures nothing on the remote backend. The price is
a process spawn per call, and it is stated as three numbers because one would be
a figure this repository's own verification disproves: on a short grade probe
(three candidates) claude cost 3.9 s and codex 8.2 s; on the lab's *real* grade
prompt claude cost 5.6 s (`grade_ms=5574.1`, 2026-08-12); and inside a judged run
its own prompts averaged ~7.4 s per call-slot. A prompt this backend is asked to
grade is longer than a probe, so the probe figure is the floor, not the price.

What makes a row produced here trustworthy is that the CLI is driven as a
completion endpoint rather than as an agent, and every flag below prevents one
specific way a row could otherwise be wrong:

* `--system-prompt` **replaces** Claude Code's agent prompt, so the stage's own
  prompt is the only instruction the model has. Measured 306 input tokens
  against codex's 18,574 for the identical request. Codex has no equivalent, so
  its instructions are prepended to the prompt body — the one place the two
  rows of `CLIS` differ in kind rather than in spelling.
* `--tools ""` / `-s read-only`. A judge holding a Read tool could grade from
  `.runs/` instead of from the context it was handed, and no field on the row
  would contradict it. The same class of fault as `llm_scores` returning 0.5
  with the daemon down: a number that looks like a measurement and is not one.
* `--setting-sources ""` / `--ignore-rules` / an empty cwd. No CLAUDE.md, no
  hooks, no skills, no project instruction files. This repository's CLAUDE.md is
  two hundred lines about which candidate won; in a judge's prompt that is an
  instruction to prefer the answer we already believe.
* `--no-session-persistence` / `--ephemeral`. The lab writes one JSON run to
  `.runs/` and one ledger row. A backend that also writes session transcripts
  writes something nobody accounted for.
* `--ignore-user-config`. The one that pays measurably: it makes a codex call
  independent of `~/.codex/config.toml`, whose `model_reasoning_effort = "high"`
  was the whole of the 17.6s-to-8.2s difference. The 18.5k-token preamble that
  remains is codex's own harness, not this machine's AGENTS.md — measured 18,574
  with the user config against 18,513 without.
* **An allowlisted environment** (`_KEEP`, `_child_env`). None of the flags above
  touch the environment, and both CLIs configure themselves from it: a row
  labelled `sonnet` that `ANTHROPIC_DEFAULT_SONNET_MODEL` pointed at another
  model is exactly the artefact the whole of this list exists to prevent.
* **UTF-8 on the decode**, explicitly rather than by the machine's locale. The
  corpus is a Farsi diary and this is the one backend whose transport is not
  HTTP+JSON, so it is the one that has to say which encoding it is reading.

**Nothing here falls back.** A non-zero exit, a timeout, an error envelope, a
reply that cannot be read, or a reply with no text at all raises `CliError`. The
empty reply is the one that had to be looked for: `model_reasoning_effort=
"minimal"` exits 0 and says nothing, and a tolerant parser reads that as "no
opinion" and scores every document 0.5 — a `grader='llm'` row measured ungated,
which is the one artefact this lab must never produce.
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

# Appended to whatever system message the stage already wrote. Measured on the
# real LLM_GRADE_PROMPT: claude's output fell from 47 tokens to 16 and codex's
# from 55 to 18, with the parsed scores unchanged in kind. An agent CLI's default
# habit is to explain itself, and every stage here parses with a regex.
API_LINE = ('You are being called as a text completion endpoint, not an '
            'assistant. Follow the instruction exactly and emit only the '
            'requested output: no preamble, no explanation, no code fences, '
            'no tool use, no questions.')


# The only variables a call is allowed to see. An **allowlist**, not a denylist:
# both CLIs read their configuration from the environment and none of the flags
# above touch it, so `ANTHROPIC_BASE_URL` (or `OPENAI_BASE_URL`) sends the call to
# a different gateway, `ANTHROPIC_DEFAULT_SONNET_MODEL` remaps the alias the row
# is labelled with, and `CLAUDE_CODE_MAX_OUTPUT_TOKENS` / `MAX_THINKING_TOKENS` /
# `CLAUDE_EFFORT` move what the argv says it fixed. All of those are present in
# the installed binary (checked by name, 2026-08-12), and a row labelled `sonnet`
# that a different model served is the one artefact this lab must never produce.
# A denylist would have to be re-read every time either CLI ships a new variable,
# and an out-of-date denylist fails silently, which is the failure this backend
# cannot have.
#
# Each entry earns its place by being needed to *reach* the model, and none of
# them can change which model answers. Verified live 2026-08-12: both CLIs
# authenticate under this list and nothing else.
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
#
# Deliberately *not* kept: `HTTPS_PROXY`, `NODE_EXTRA_CA_CERTS`, `SSL_CERT_FILE`
# and their kin. A machine that needs one gets a call that cannot connect, which
# raises `CliError` and says so — the opposite of a redirected base url, which
# answers.


def _child_env() -> dict[str, str]:
    """The environment one call runs with: `_KEEP`, and nothing else."""
    return {name: value for name, value in os.environ.items()
            if name in _KEEP or name.startswith(_KEEP_PREFIX)}


class CliError(RuntimeError):
    """A CLI backend did not produce a reply this lab can use.

    Deliberately one class for every failure: `retrieval.llm_scores` turns any
    exception into `GradeUnavailable` and `/api/queries` answers 502 naming the
    stage, so what the caller needs is that it failed and a sentence saying how.
    """


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

    `stop_reason` is read for one value only. A reply cut off at the output limit
    parses exactly like a complete one — a grade list of eight that stops at four
    leaves `llm_scores` reading the four it never reached as *no opinion* and
    scoring them 0.5, which clears the gate's 0.4 threshold. That is the
    empty-reply rule with the reply half-arrived, so it gets the same answer.
    Only `max_tokens` is refused, rather than everything that is not `end_turn`:
    this guard, like `provider_problems`, refuses what it has verified and not
    what it fails to recognise, and a stop reason a later version ships is not a
    truncation.
    """
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
    """The last agent message, and the usage the turn ended with.

    The *last*, not the first: a turn that says anything before its answer would
    otherwise be scored instead of the answer. Unparseable lines are skipped
    rather than fatal — the stream is a log, and a log gaining a line the lab
    does not know about must not end a judged run.
    """
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
    """Whether this machine has the command.

    The whole availability check for a CLI backend, and deliberately so: an
    Ollama tag can be asked of `/api/tags`, but a CLI alias cannot be checked
    without paying for a call. So the fact verified is the one that *can* be —
    and an alias this account cannot use fails at call time with the CLI's own
    message, which is a better error than anything guessed here.
    """
    spec = CLIS.get(cli)
    return bool(spec and shutil.which(spec.binary))


def checked_effort(cli: str, effort: str) -> str:
    """The effort value, or a refusal naming what this CLI accepts.

    The two do not agree — claude takes 'xhigh', codex takes 'none' — and codex
    answers an unaccepted value with exit 0 and no text at all. Passed through,
    that is a stage which scored every document 0.5; refused here, it is a
    sentence saying what to write instead.
    """
    spec = CLIS[cli]
    if effort not in spec.efforts:
        raise ValueError(
            f'{effort!r} is not a reasoning effort {spec.binary} accepts; '
            'expected one of ' + ', '.join(repr(v) for v in spec.efforts)
            + '. Measured: an unaccepted value exits 0 and answers nothing, so '
            'this refuses rather than letting a stage score everything 0.5')
    return effort


def _instructions_and_body(messages: list[BaseMessage]) -> tuple[str, str]:
    """The system text and everything else, flattened.

    Every LLM call in this lab is single-turn — one system prompt, one user
    message — so "everything else" is one message in practice. An assistant turn
    is labelled rather than silently merged, because a CLI reading it as its own
    instruction would be answering a different request.
    """
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
    """Unwrap a reply that is *entirely* one code fence, and nothing else.

    A chat-tuned CLI fences a block of scores out of habit; that is a property
    of the transport, not a different measurement, so it is undone. A reply that
    merely contains a fence is returned byte-exact: cleaning those up per
    backend is how a row measured on claude stops being comparable with a row
    measured on ollama.
    """
    match = _WHOLE_FENCE.match(text)
    if not match or '```' in match.group('body'):
        return text
    return match.group('body').strip()


def _run(spec: CliSpec, argv: list[str], prompt: str, timeout: int) -> tuple[str, dict]:
    """One call. Every way it can fail raises, and each says which."""
    # A fresh directory per call, not one per process: it costs a mkdir against
    # a call that already costs seconds, and it means nothing one call leaves
    # behind can be read by the next.
    with tempfile.TemporaryDirectory() as empty:
        try:
            # `encoding='utf-8'` rather than `text=True`, which decodes with
            # `locale.getpreferredencoding(False)`: under a C/POSIX locale — a
            # launchd job, cron, a CI shell — that is ASCII, and this corpus is a
            # Farsi diary, so a correct answer would fail to decode; under a
            # latin-1 locale it would not fail at all and RAGAS would score
            # mojibake with confidence. Every other backend gets UTF-8 fixed for
            # it by HTTP+JSON. `errors='strict'` for the reason nothing else here
            # falls back: replacing an undecodable byte changes the text that was
            # scored and leaves no field on the row saying so.
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
            # Every way this can fail says which, and this one would otherwise
            # leave `_run` as neither a CliError nor a reply.
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

    Sync only. RAGAS drives its judge asynchronously, and `BaseChatModel`'s
    default `_agenerate` runs `_generate` in a thread — which is the right shape
    here anyway, since the thing being parallelised is a process rather than a
    socket (`ragas_eval.JUDGE_LOAD` caps it at three).

    No `bind_tools`. `FakeChat` returns itself from it, which is harmless for a
    fake and would be a lie here: this backend cannot do tool calling, and
    saying so is better than a stage silently getting no tool call.
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
