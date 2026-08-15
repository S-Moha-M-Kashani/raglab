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

# Kept verbatim rather than minimised: the fields this lab reads are the
# fields it must keep reading when the CLI's envelope grows others.
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

    Records the call so the argv and cwd can be asserted. Does *not* list the
    working directory's contents — that costs an extra syscall per call and
    only one test needs it, so that test stubs it inline instead.
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
                           'errors': kwargs.get('errors')})
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


def test_a_claude_call_is_driven_as_a_completion_endpoint_not_an_agent(monkeypatch):
    """`--system-prompt` replaces Claude Code's agent prompt so the stage's own
    prompt is the only instruction; `--tools ""` means a judge cannot go read
    `.runs/` instead of the context it was handed."""
    # this is a unit test
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


def test_each_call_runs_isolated_outside_this_repo_with_a_scrubbed_environment(
        monkeypatch):
    """Both CLIs discover instruction files from the working directory —
    pointed at a directory holding the previous call's leftovers, call N
    could read call N-1. And the flags scrub settings, config, rules, cwd and
    tools, but none of them touch the environment, which both CLIs also read
    configuration from — `ANTHROPIC_DEFAULT_SONNET_MODEL` could remap the
    alias a row is labelled with. An allowlist rather than a denylist, since a
    denylist goes stale the moment either CLI ships a new variable."""
    # this is a unit test
    monkeypatch.setenv('ANTHROPIC_BASE_URL', 'https://elsewhere.example')
    monkeypatch.setenv('ANTHROPIC_DEFAULT_SONNET_MODEL', 'some-other-model')
    monkeypatch.setenv('CLAUDE_CODE_MAX_OUTPUT_TOKENS', '32')
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://elsewhere.example')
    monkeypatch.setenv('CLAUDE_EFFORT', 'max')
    for cli, stdout in (('claude', CLAUDE_ENVELOPE), ('codex', CODEX_EVENTS)):
        seen = {}

        def stub(argv, cli=cli, stdout=stdout, **kwargs):
            # The one test that needs the directory's contents — listed
            # *during* the call, since the real cwd is a TemporaryDirectory
            # gone by the time an assertion could look at it.
            seen['contents'] = sorted(os.listdir(kwargs['cwd']))
            seen['cwd'] = kwargs['cwd']
            seen['env'] = kwargs.get('env')
            return subprocess.CompletedProcess(argv, 0, stdout, '')

        monkeypatch.setattr(clichat.subprocess, 'run', stub)
        model = clichat.CliChat(cli=cli, model='sonnet' if cli == 'claude'
                                else 'gpt-5.6-terra')
        model.invoke([{'role': 'system', 'content': SYSTEM},
                     {'role': 'user', 'content': BODY}])
        assert seen['contents'] == []
        assert ROOT not in Path(seen['cwd']).parents

        env = seen['env']
        assert env is not None, 'inheriting os.environ is the defect'
        assert not [name for name in env
                    if name.startswith(('ANTHROPIC_', 'OPENAI_', 'CLAUDE_CODE_'))
                    or name == 'CLAUDE_EFFORT'], env
        # …and it still carries what the CLI needs to reach the model at all:
        # PATH to find the command, HOME to find the login it is already using.
        assert env['PATH'] == os.environ['PATH']
        assert env['HOME'] == os.environ['HOME']


def test_the_reply_is_decoded_as_utf8_rather_than_as_the_machine_asked(monkeypatch):
    """`text=True` decodes with the preferred locale encoding, which under a
    C/POSIX locale is ASCII — a correct Farsi answer would raise there.
    Every other backend gets UTF-8 fixed by HTTP+JSON; this one's transport
    is bytes on a pipe."""
    # this is a unit test
    farsi = 'دیروز برنج خوردم'
    stdout = json.dumps({'is_error': False, 'usage': {}, 'result': farsi})
    message, recorder = reply(monkeypatch, 'claude', stdout=stdout)
    assert message.content == farsi
    call = recorder.calls[0]
    assert call['encoding'] == 'utf-8'
    # Loud rather than lossy: a replaced byte is a changed measurement with no
    # field on the row saying it changed.
    assert call['errors'] == 'strict'


def test_a_codex_call_carries_its_instructions_in_the_prompt(monkeypatch):
    """Codex has no flag that replaces its system prompt, so the stage's text
    is prepended to the prompt body instead. `--ignore-user-config` is what
    makes the call reproducible against this machine's own config.toml."""
    # this is a unit test
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


def test_a_recorded_claude_envelope_parses_to_its_text_and_its_usage(monkeypatch):
    """The reply is the envelope's `result`, byte-exact, since the stage that
    asked for it parses it with a regex written for the other backends."""
    # this is a unit test
    message, _ = reply(monkeypatch, 'claude', stdout=CLAUDE_ENVELOPE)
    assert message.content == '1: 8\n2: 0\n3: 8'
    assert message.usage_metadata['input_tokens'] == 369
    assert message.usage_metadata['output_tokens'] == 16
    assert message.usage_metadata['total_tokens'] == 385


def test_a_recorded_codex_stream_parses_to_its_last_message_and_its_usage(monkeypatch):
    """Codex answers as a JSONL event stream; reading the *last*
    agent_message matters because a turn can say something before its
    answer."""
    # this is a unit test
    message, _ = reply(monkeypatch, 'codex', stdout=CODEX_EVENTS)
    assert message.content == '1: 10\n2: 0\n3: 10'
    assert message.usage_metadata['input_tokens'] == 18513
    assert message.usage_metadata['output_tokens'] == 18


def test_a_reply_that_is_entirely_one_fence_is_unwrapped_and_nothing_else_is(monkeypatch):
    """A chat-tuned CLI fences a block of scores out of habit; that is a
    property of the transport, not a different measurement, so it is undone.
    A reply that merely *contains* a fence is left byte-exact."""
    # this is a unit test
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

    # Opens and closes with a fence but holds one *inside*: not one fenced
    # block, however much it looks like one to an anchored regex.
    nested = json.dumps({'is_error': False, 'usage': {},
                         'result': '```\n1: 8\n```\nand\n```\n2: 0\n```'})
    message, _ = reply(monkeypatch, 'claude', stdout=nested)
    assert message.content == '```\n1: 8\n```\nand\n```\n2: 0\n```'


def test_a_reply_cut_off_at_the_output_limit_is_refused_not_scored(monkeypatch):
    """A truncated reply parses exactly like a complete one, silently scoring
    the unreached part as "no opinion". Only `max_tokens` is refused: a stop
    reason a later version ships is not a known truncation."""
    # this is a unit test
    cut = json.dumps({'is_error': False, 'stop_reason': 'max_tokens',
                      'usage': {'input_tokens': 369, 'output_tokens': 4},
                      'result': '1: 8\n2: 0'})
    with pytest.raises(clichat.CliError, match='output limit'):
        reply(monkeypatch, 'claude', stdout=cut)
    message, _ = reply(monkeypatch, 'claude', stdout=CLAUDE_ENVELOPE)
    assert message.content == '1: 8\n2: 0\n3: 8'


FAULT_CASES = [
    ('claude', dict(boom=UnicodeDecodeError('utf-8', b'\xff', 0, 1,
                                            'invalid start byte')),
     'not UTF-8'),
    ('claude', dict(stdout=json.dumps({'is_error': False, 'usage': {},
                                       'result': ''})),
     'no text'),
    ('codex', dict(stdout=json.dumps({'type': 'turn.completed', 'usage': {}})),
     'no text'),
    ('claude', dict(stdout='', returncode=1, stderr='nope'), 'exited 1'),
    ('claude', dict(stdout=json.dumps({'is_error': True,
                                       'result': 'Credit balance too low'})),
     'Credit balance'),
    ('claude', dict(stdout='not json at all'), 'cannot read'),
    ('claude', dict(boom=FileNotFoundError('claude')), 'not installed'),
    ('claude', dict(boom=subprocess.TimeoutExpired(cmd='claude', timeout=600)),
     'did not answer'),
]


@pytest.mark.parametrize('cli, kwargs, expected', FAULT_CASES, ids=[
    'undecodable-bytes', 'empty-reply-claude', 'empty-reply-codex',
    'nonzero-exit', 'error-envelope', 'unreadable-stdout',
    'missing-command', 'timeout'])
def test_a_cli_fault_raises_a_named_cli_error(monkeypatch, cli, kwargs, expected):
    """One contract for every way this backend can fail: raise `CliError`
    naming what went wrong, rather than letting a bad reply score as if it
    were a real measurement. Eight faults, one rule — a decode error, an
    empty reply from either CLI, a non-zero exit, an error envelope,
    unreadable stdout, a missing command and a timeout."""
    # this is a unit test
    with pytest.raises(clichat.CliError, match=expected):
        reply(monkeypatch, cli, **kwargs)


def test_an_effort_the_cli_does_not_accept_is_refused_before_any_call():
    """The two CLIs accept different values — claude takes 'xhigh', codex
    takes 'none' — and codex answers an unaccepted one with exit 0 and no
    text, so the check must happen before the call."""
    # this is a unit test
    assert clichat.checked_effort('claude', 'low') == 'low'
    assert clichat.checked_effort('codex', 'none') == 'none'
    with pytest.raises(ValueError, match='xhigh'):
        clichat.checked_effort('codex', 'xhigh')
    with pytest.raises(ValueError, match='minimal'):
        clichat.checked_effort('claude', 'minimal')


def test_availability_is_the_binary_because_there_is_nothing_else_to_ask(monkeypatch):
    """A CLI alias cannot be checked without paying for a call, so this
    verifies the one thing it can: whether the command is on this machine."""
    # this is a unit test
    monkeypatch.setattr(clichat.shutil, 'which',
                        lambda name: '/usr/bin/claude' if name == 'claude' else None)
    assert clichat.cli_available('claude') is True
    assert clichat.cli_available('codex') is False
    assert clichat.cli_available('wat') is False
