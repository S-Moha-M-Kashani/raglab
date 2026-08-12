"""The lab's third kind of backend: a CLI on this machine, run as a subprocess.

Every test here stubs `subprocess.run`. The suite is offline and must stay that
way, and a test that spawned a real `claude` would also be a test whose result
depended on a login.
"""
import json
import os
import subprocess
from pathlib import Path

import pytest

from raglab import clichat

ROOT = Path(__file__).resolve().parents[1]

# The stage prompt these were recorded against, shortened. The point is that a
# system message exists and has to arrive somewhere the model reads it.
SYSTEM = 'You score how useful each numbered excerpt is. Reply "<n>: <0-10>".'
BODY = 'Question: What did I eat?\n\n[1] I had rice for lunch today.'

# Recorded from a real call on 2026-08-12 (see the design doc). Kept verbatim
# rather than minimised: the fields this lab reads are the fields it must keep
# reading when the CLI's envelope grows others.
CLAUDE_ENVELOPE = json.dumps({
    'is_error': False, 'subtype': 'success', 'num_turns': 1,
    'stop_reason': 'end_turn', 'total_cost_usd': 0.001672,
    'usage': {'input_tokens': 369, 'output_tokens': 16,
              'cache_read_input_tokens': 0, 'cache_creation_input_tokens': 0},
    'result': '1: 8\n2: 0\n3: 8', 'type': 'result'})

CODEX_EVENTS = '\n'.join([
    json.dumps({'type': 'thread.started', 'thread_id': '019ff52a-3d6c'}),
    json.dumps({'type': 'turn.started'}),
    json.dumps({'type': 'item.completed',
                'item': {'id': 'item_0', 'type': 'agent_message',
                         'text': '1: 10\n2: 0\n3: 10'}}),
    json.dumps({'type': 'turn.completed',
                'usage': {'input_tokens': 18513, 'cached_input_tokens': 6912,
                          'output_tokens': 18, 'reasoning_output_tokens': 0}}),
])


class Recorder:
    """Stands in for the CLI.

    Records the call so the argv can be asserted, and lists the working
    directory *during* the call — the real one is a TemporaryDirectory that is
    gone by the time an assertion could look at it.
    """

    def __init__(self, stdout='', returncode=0, stderr='', boom=None):
        self.stdout, self.returncode, self.stderr = stdout, returncode, stderr
        self.boom = boom
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append({'argv': argv, 'input': kwargs.get('input'),
                           'cwd': kwargs.get('cwd'),
                           'timeout': kwargs.get('timeout'),
                           'env': kwargs.get('env'),
                           'encoding': kwargs.get('encoding'),
                           'errors': kwargs.get('errors'),
                           'contents': sorted(os.listdir(kwargs['cwd']))})
        if self.boom is not None:
            raise self.boom
        return subprocess.CompletedProcess(argv, self.returncode,
                                           self.stdout, self.stderr)

    @property
    def argv(self):
        return self.calls[0]['argv']


def reply(monkeypatch, cli, **kwargs):
    """Run one call against a stubbed CLI and hand back (message, recorder)."""
    recorder = Recorder(**kwargs)
    monkeypatch.setattr(clichat.subprocess, 'run', recorder)
    model = clichat.CliChat(cli=cli, model='sonnet' if cli == 'claude'
                            else 'gpt-5.6-terra')
    message = model.invoke([{'role': 'system', 'content': SYSTEM},
                            {'role': 'user', 'content': BODY}])
    return message, recorder


# This is a unit test.
def test_a_claude_call_is_driven_as_a_completion_endpoint_not_an_agent(monkeypatch):
    """The whole reason this backend is trustworthy. `--system-prompt` replaces
    Claude Code's agent prompt, so the stage's own prompt is the only
    instruction; `--tools ""` means a judge cannot go and read `.runs/` instead
    of the context it was handed; and the empty cwd with no setting sources
    keeps this repository's CLAUDE.md — two hundred lines about which candidate
    won — out of a judge's prompt.
    """
    _, recorder = reply(monkeypatch, 'claude', stdout=CLAUDE_ENVELOPE)
    argv = recorder.argv
    assert argv[0] == 'claude' and '-p' in argv
    assert argv[argv.index('--output-format') + 1] == 'json'
    assert argv[argv.index('--model') + 1] == 'sonnet'
    # A tool is not merely unnecessary here, it is a way for a score to be wrong.
    assert argv[argv.index('--tools') + 1] == ''
    assert argv[argv.index('--setting-sources') + 1] == ''
    assert '--no-session-persistence' in argv
    assert '--strict-mcp-config' in argv
    # The stage's system text and the completion-endpoint line both reach the
    # model, and they reach it as the system prompt rather than as user text.
    instructions = argv[argv.index('--system-prompt') + 1]
    assert SYSTEM in instructions and clichat.API_LINE in instructions
    # …so stdin is the question alone.
    assert recorder.calls[0]['input'].strip() == BODY


# This is a unit test.
def test_the_cli_runs_in_an_empty_directory_outside_this_repository(monkeypatch):
    """Both CLIs discover instruction files from the working directory. Pointed
    at the lab, a judge would read the repository's own account of which
    candidate won; pointed at a directory holding the previous call's leftovers,
    call N could read call N-1. So it is a fresh empty directory each time."""
    for cli, stdout in (('claude', CLAUDE_ENVELOPE), ('codex', CODEX_EVENTS)):
        _, recorder = reply(monkeypatch, cli, stdout=stdout)
        call = recorder.calls[0]
        assert call['contents'] == []
        assert ROOT not in Path(call['cwd']).parents


# This is a unit test.
def test_a_call_sees_an_allowlisted_environment_and_not_this_shell(monkeypatch):
    """The flags scrub settings files, user config, rules, cwd and tools — and
    none of them touches the environment, which is where both CLIs also read
    their configuration. `ANTHROPIC_BASE_URL` sends the call to another gateway
    and `ANTHROPIC_DEFAULT_SONNET_MODEL` remaps the alias the row is labelled
    with, so a row saying `sonnet` would name a model that never answered. An
    allowlist rather than a denylist, because a denylist goes out of date every
    time either CLI ships a variable and does it silently."""
    monkeypatch.setenv('ANTHROPIC_BASE_URL', 'https://elsewhere.example')
    monkeypatch.setenv('ANTHROPIC_DEFAULT_SONNET_MODEL', 'some-other-model')
    monkeypatch.setenv('CLAUDE_CODE_MAX_OUTPUT_TOKENS', '32')
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://elsewhere.example')
    monkeypatch.setenv('CLAUDE_EFFORT', 'max')
    for cli, stdout in (('claude', CLAUDE_ENVELOPE), ('codex', CODEX_EVENTS)):
        _, recorder = reply(monkeypatch, cli, stdout=stdout)
        env = recorder.calls[0]['env']
        assert env is not None, 'inheriting os.environ is the defect'
        assert not [name for name in env
                    if name.startswith(('ANTHROPIC_', 'OPENAI_', 'CLAUDE_CODE_'))
                    or name == 'CLAUDE_EFFORT'], env
        # …and it still carries what the CLI needs to reach the model at all:
        # PATH to find the command, HOME to find the login it is already using.
        assert env['PATH'] == os.environ['PATH']
        assert env['HOME'] == os.environ['HOME']


# This is a unit test.
def test_the_reply_is_decoded_as_utf8_rather_than_as_the_machine_asked(monkeypatch):
    """`text=True` decodes with the preferred locale encoding, which under a
    C/POSIX locale is ASCII — so a correct Farsi answer would raise, and under a
    latin-1 locale it would not raise at all and RAGAS would score mojibake with
    confidence. Every other backend has UTF-8 fixed for it by HTTP+JSON; this is
    the one whose transport is bytes on a pipe."""
    farsi = 'دیروز برنج خوردم'
    stdout = json.dumps({'is_error': False, 'usage': {}, 'result': farsi})
    message, recorder = reply(monkeypatch, 'claude', stdout=stdout)
    assert message.content == farsi
    call = recorder.calls[0]
    assert call['encoding'] == 'utf-8'
    # Loud rather than lossy: a replaced byte is a changed measurement with no
    # field on the row saying it changed.
    assert call['errors'] == 'strict'


# This is a unit test.
def test_undecodable_bytes_are_a_named_failure_and_not_a_bare_one(monkeypatch):
    """A UnicodeDecodeError raised inside `subprocess.run` used to escape `_run`
    as neither a reply nor a CliError, so the one contract this module has —
    every failure is a CliError saying how — had a hole in it exactly where a
    Farsi corpus would find it."""
    boom = UnicodeDecodeError('utf-8', b'\xff', 0, 1, 'invalid start byte')
    with pytest.raises(clichat.CliError, match='not UTF-8'):
        reply(monkeypatch, 'claude', boom=boom)


# This is a unit test.
def test_a_codex_call_carries_its_instructions_in_the_prompt(monkeypatch):
    """Codex has no flag that replaces its system prompt, so the stage's text is
    prepended to the prompt body instead — the one place the two rows of CLIS
    differ in kind rather than in spelling. `--ignore-user-config` is what makes
    the call reproducible: this machine's config.toml asked for high reasoning
    effort, which was the whole of the 17.6s-to-8.2s difference."""
    _, recorder = reply(monkeypatch, 'codex', stdout=CODEX_EVENTS)
    argv = recorder.argv
    assert argv[:2] == ['codex', 'exec'] and '--json' in argv
    assert '--system-prompt' not in argv
    assert '--ignore-user-config' in argv and '--ephemeral' in argv
    assert argv[argv.index('-s') + 1] == 'read-only'
    assert argv[argv.index('-m') + 1] == 'gpt-5.6-terra'
    assert 'model_reasoning_effort="low"' in argv
    prompt = recorder.calls[0]['input']
    assert prompt.startswith(SYSTEM) or clichat.API_LINE in prompt.split(BODY)[0]
    assert BODY in prompt


# This is a unit test.
def test_a_recorded_claude_envelope_parses_to_its_text_and_its_usage(monkeypatch):
    """The reply the lab uses is the envelope's `result`, byte-exact, because
    the stage that asked for it parses it with a regex written for the other
    backends. Usage is carried too: a backend reporting none would leave the
    token-and-cost path unexercised, which is half of why FakeChat exists."""
    message, _ = reply(monkeypatch, 'claude', stdout=CLAUDE_ENVELOPE)
    assert message.content == '1: 8\n2: 0\n3: 8'
    assert message.usage_metadata['input_tokens'] == 369
    assert message.usage_metadata['output_tokens'] == 16
    assert message.usage_metadata['total_tokens'] == 385


# This is a unit test.
def test_a_recorded_codex_stream_parses_to_its_last_message_and_its_usage(monkeypatch):
    """Codex answers as a JSONL event stream, so the reply is the last
    agent_message and the usage is the turn.completed event. Reading the *last*
    one matters: a turn that says anything before its answer would otherwise be
    scored instead of the answer."""
    message, _ = reply(monkeypatch, 'codex', stdout=CODEX_EVENTS)
    assert message.content == '1: 10\n2: 0\n3: 10'
    assert message.usage_metadata['input_tokens'] == 18513
    assert message.usage_metadata['output_tokens'] == 18


# This is a unit test.
def test_a_reply_that_is_entirely_one_fence_is_unwrapped_and_nothing_else_is(monkeypatch):
    """A chat-tuned CLI fences a block of scores out of habit; that is a
    property of the transport, not a different measurement, so it is undone. A
    reply that merely *contains* a fence is left byte-exact — cleaning that up
    per backend is how a row measured on claude stops being comparable with a
    row measured on ollama."""
    fenced = json.dumps({'is_error': False, 'usage': {},
                         'result': '```\n1: 8\n2: 0\n```'})
    message, _ = reply(monkeypatch, 'claude', stdout=fenced)
    assert message.content == '1: 8\n2: 0'

    tagged = json.dumps({'is_error': False, 'usage': {},
                         'result': '```json\n{"a": 1}\n```'})
    message, _ = reply(monkeypatch, 'claude', stdout=tagged)
    assert message.content == '{"a": 1}'

    prose = json.dumps({'is_error': False, 'usage': {},
                        'result': 'scores:\n```\n1: 8\n```\nand that is all'})
    message, _ = reply(monkeypatch, 'claude', stdout=prose)
    assert message.content == 'scores:\n```\n1: 8\n```\nand that is all'

    # A reply that opens and closes with a fence but holds one *inside* is not
    # one fenced block, however much it looks like one to an anchored regex —
    # unwrapping it would hand the stage a stray fence line in the middle of the
    # scores and leave nothing on the row saying the text had been edited.
    nested = json.dumps({'is_error': False, 'usage': {},
                         'result': '```\n1: 8\n```\nand\n```\n2: 0\n```'})
    message, _ = reply(monkeypatch, 'claude', stdout=nested)
    assert message.content == '```\n1: 8\n```\nand\n```\n2: 0\n```'


# This is a unit test.
def test_a_reply_cut_off_at_the_output_limit_is_refused_not_scored(monkeypatch):
    """A truncated reply parses exactly like a complete one, and a grade list of
    eight that stops at four leaves `llm_scores` reading the four it never
    reached as "no opinion" and scoring them 0.5 — which clears the gate's 0.4
    threshold. Same argument as the empty reply, with the reply half-arrived.
    Only `max_tokens` is refused: a stop reason a later version ships is not a
    truncation, and this guard refuses what it has verified."""
    cut = json.dumps({'is_error': False, 'stop_reason': 'max_tokens',
                      'usage': {'input_tokens': 369, 'output_tokens': 4},
                      'result': '1: 8\n2: 0'})
    with pytest.raises(clichat.CliError, match='output limit'):
        reply(monkeypatch, 'claude', stdout=cut)
    # And the ordinary end is still ordinary — CLAUDE_ENVELOPE carries
    # stop_reason='end_turn', which every other test here relies on.
    message, _ = reply(monkeypatch, 'claude', stdout=CLAUDE_ENVELOPE)
    assert message.content == '1: 8\n2: 0\n3: 8'


# This is a unit test.
def test_an_empty_reply_raises_instead_of_reaching_a_tolerant_parser(monkeypatch):
    """Measured, not imagined: `model_reasoning_effort="minimal"` exits 0 and
    says nothing. `retrieval.llm_scores` reads an unparsed line as "no opinion"
    and scores it 0.5, which clears the gate's 0.4 threshold — so a silent empty
    reply is a `grader='llm'` row that was measured ungated, and no field on it
    would say so."""
    empty = json.dumps({'is_error': False, 'usage': {}, 'result': ''})
    with pytest.raises(clichat.CliError, match='no text'):
        reply(monkeypatch, 'claude', stdout=empty)
    with pytest.raises(clichat.CliError, match='no text'):
        reply(monkeypatch, 'codex', stdout=json.dumps({'type': 'turn.completed',
                                                       'usage': {}}))


# This is a unit test.
def test_a_failed_call_raises_and_says_which_command_failed(monkeypatch):
    """Three failures, one rule: the lab would rather stop than score. A backend
    that swallowed these would produce numbers indistinguishable from a
    measurement — the fault `llm_scores` was fixed for on 2026-08-02."""
    with pytest.raises(clichat.CliError, match='exited 1'):
        reply(monkeypatch, 'claude', stdout='', returncode=1, stderr='nope')
    errored = json.dumps({'is_error': True, 'result': 'Credit balance too low'})
    with pytest.raises(clichat.CliError, match='Credit balance'):
        reply(monkeypatch, 'claude', stdout=errored)
    with pytest.raises(clichat.CliError, match='cannot read'):
        reply(monkeypatch, 'claude', stdout='not json at all')


# This is a unit test.
def test_a_missing_command_and_a_timeout_are_named_rather_than_bare(monkeypatch):
    """The two failures a user can act on. "claude is not installed" and "claude
    did not answer in 600s" are different problems with different fixes, and a
    bare OSError says neither."""
    with pytest.raises(clichat.CliError, match='not installed'):
        reply(monkeypatch, 'claude', boom=FileNotFoundError('claude'))
    with pytest.raises(clichat.CliError, match='did not answer'):
        reply(monkeypatch, 'claude',
              boom=subprocess.TimeoutExpired(cmd='claude', timeout=600))


# This is a unit test.
def test_a_per_role_model_is_forwarded_per_call(monkeypatch):
    """The lab's convention: one client serves every stage and a named model
    rides on the request (models.ROLES). For a subprocess that means the argv,
    so a run whose reranker and judge differ must produce two different argv."""
    recorder = Recorder(stdout=CLAUDE_ENVELOPE)
    monkeypatch.setattr(clichat.subprocess, 'run', recorder)
    model = clichat.CliChat(cli='claude', model='sonnet')
    model.invoke([{'role': 'user', 'content': BODY}], model='opus')
    assert recorder.argv[recorder.argv.index('--model') + 1] == 'opus'


# This is a unit test.
def test_an_effort_the_cli_does_not_accept_is_refused_before_any_call():
    """The two CLIs accept different values — claude takes 'xhigh', codex takes
    'none' — and codex answers an unaccepted one with exit 0 and no text. So the
    check happens here, naming what the CLI does accept, rather than becoming a
    stage that scored everything 0.5."""
    assert clichat.checked_effort('claude', 'low') == 'low'
    assert clichat.checked_effort('codex', 'none') == 'none'
    with pytest.raises(ValueError, match='xhigh'):
        clichat.checked_effort('codex', 'xhigh')
    with pytest.raises(ValueError, match='minimal'):
        clichat.checked_effort('claude', 'minimal')


# This is a unit test.
def test_availability_is_the_binary_because_there_is_nothing_else_to_ask(monkeypatch):
    """An Ollama tag can be checked against /api/tags. A CLI alias cannot be
    checked at all without paying for a call, so the fact this lab verifies is
    the one it can: whether the command is on this machine."""
    monkeypatch.setattr(clichat.shutil, 'which',
                        lambda name: '/usr/bin/claude' if name == 'claude' else None)
    assert clichat.cli_available('claude') is True
    assert clichat.cli_available('codex') is False
    assert clichat.cli_available('wat') is False
