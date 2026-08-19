"""The skills tools' live probe — collected everywhere, run only when named.

It pays for a real CLI call, which the suite may never do silently — so the
guard is invocation-shaped: the test runs only when this file (or the test
itself) was named on the pytest command line, and skips under any broad
sweep (`pytest tests/`, a merge gate, `uv run pytest`). VS Code's play
button names the test's node id, so it runs there; discovery still sees it,
which is what puts the button beside the function at all.

    uv run pytest test_skills_live.py -v -s

`test_skills.py` holds the offline coverage; this file holds the one
claim only a real model can settle — that the search_rag_skills docstring,
which is the whole prompt a tool-running model gets, actually steers one.
"""
import re

import pytest

from raglab.agents.widget import skills_corpus_loader as skills
from raglab.agents import widget
from raglab.llm_backends.cli_subprocess_chat import CliChat, cli_available
from raglab.configuration.env_settings import load_env_file

BILINGUAL_QUESTION = ('I would like an architecture for a diary book with '
                      'English and Farsi content, where I need an embedding '
                      'that searches across both. Which embedding method do '
                      'you recommend?')


def _named_on_the_command_line(request) -> bool:
    """Whether this run asked for this file by name, as the play button and
    an explicit terminal run do — a directory or bare sweep did not."""
    return any('test_skills_live' in arg
               for arg in request.config.invocation_params.args)


def test_a_cli_model_picks_keywords_that_find_the_multilingual_skill(request):
    # this is an integration test
    """End to end: shown a bilingual-corpus question, the model chooses its
    own keywords, and those keywords, run through the real search, must find
    `multilingual-rag` (the skill that owns encoder choice for non-English
    text). One call on the lightest codex alias (`gpt-5.6-luna`, effort low
    — the membership is finite)."""
    if not _named_on_the_command_line(request):
        pytest.skip('a real codex call: run this file by name to opt in')
    if not cli_available('codex'):
        pytest.skip('the codex CLI is not on PATH')
    from langchain_core.messages import HumanMessage, SystemMessage
    system = (
        'You are an agent with a tool named search_rag_skills. Its '
        'documentation:\n\n' + widget.search_rag_skills.description
        + '\n\nReply with exactly one line, the tool call you would make, '
          'in the form: search_rag_skills(query="...")')
    question = f'The user asked: "{BILINGUAL_QUESTION}" What is your next action?'
    chat = CliChat(cli='codex', model='gpt-5.6-luna', effort='low')
    reply = chat.invoke([SystemMessage(system), HumanMessage(question)]).content
    print('\n--- the model\'s full reply ---')
    print(reply)
    print('--- end of reply ---')
    assert 'search_rag_skills' in reply
    matched = re.search(r'query="([^"]+)"', reply)
    assert matched, f'no parseable query in the reply: {reply!r}'
    found = [name for name, _ in skills.search(matched.group(1))]
    print(f'the model searched {matched.group(1)!r}; found: {found}')
    assert 'multilingual-rag' in found, (
        f'the model searched {matched.group(1)!r}, which found {found}')


@pytest.mark.parametrize('model', ['openai/gpt-5-nano', 'codex', 'claude'])
def test_the_widget_answers_the_bilingual_question_end_to_end(
        request, monkeypatch, model):
    # this is an end-to-end test
    """The full widget: the real entry point (`ask`), the shipped prompt and
    middleware — one bilingual question in, a final answer out, the run's
    account printed from HOOK_LOG. Three backends, because they take two
    different roads and both must arrive: the OpenRouter model runs the tool
    loop, so its trajectory must show search_rag_skills fired; the two CLIs
    cannot run tools (`CliChat` has no `bind_tools`) and answer in one call
    with the skills index inlined, so for them only the two agent-level
    hooks and the answer itself can be asserted.

    The conftest session guard blanks the developer's key and LangSmith
    variables precisely so no test reads them *silently* — and `.env` is
    read directly here, over that guard, because this test is the stated
    opposite: opt-in, loud, and about nothing but the real call. The widget
    is also the one module allowed to phone LangSmith, so tracing this run
    is its designed behaviour, not a leak. `monkeypatch` restores the blanks
    afterwards."""
    if not _named_on_the_command_line(request):
        pytest.skip('a real LLM call: run this file by name to opt in')
    kind, _ = widget.WIDGET_MODELS[model]
    if kind == 'cli':
        if not cli_available(model):
            pytest.skip(f'the {model} CLI is not on PATH')
    else:
        from raglab.configuration.env_settings import ROOT
        env_file = ROOT / '.env'
        values = {}
        if env_file.exists():
            for line in env_file.read_text(encoding='utf-8').splitlines():
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    name, _, value = line.partition('=')
                    values[name.strip()] = value.strip()
        missing = [name for name in widget.REQUIRED_ENV
                   if not values.get(name, '').strip()]
        if missing:
            pytest.skip('not set in .env: ' + ', '.join(missing))
        for name in widget.REQUIRED_ENV:
            monkeypatch.setenv(name, values[name])
        load_env_file()                  # the rest, e.g. OPENROUTER_BASE_URL
    widget.reset()
    widget.HOOK_LOG.clear()
    try:
        reply = widget.ask(BILINGUAL_QUESTION, model=model)
    finally:
        widget.reset()   # no cached agent may outlive the real key
        print(f'\n--- {model}: the hook log, the whole run in order ---')
        for line in widget.HOOK_LOG:
            print(line)
        print('--- end of hook log ---')
    print('--- the final answer ---')
    print(reply)
    print('--- end ---')
    assert widget.HOOK_LOG[0].startswith('before_agent'), (
        'the run did not open through the before_agent hook')
    assert widget.HOOK_LOG[-1].startswith('after_agent'), (
        'the run did not close through the after_agent hook')
    assert reply.strip(), 'the widget returned an empty answer'
    assert re.search(r'multilingual|embedding', reply, re.I), (
        f'the answer does not speak to the question: {reply!r}')
    if kind == 'openrouter':
        # Either road is grounded: the skills corpus (field knowledge) or
        # the bilingual probe (a live measurement) — answering from neither
        # is the failure. The prompt offers both for this question.
        tool_calls = [line for line in widget.HOOK_LOG
                      if line.startswith('wrap_tool_call')]
        assert any('search_rag_skills' in line
                   or 'measure_bilingual_alignment' in line
                   for line in tool_calls), (
            f'the model used no grounded source; tool calls: {tool_calls}')


def test_the_bilingual_probe_measures_the_default_encoder_as_aligned(request):
    # this is an integration test
    """The real measurement behind the fake-encoder unit tests: the lab's
    default encoder, `heydariAI/persian-embeddings`, measured aligned on
    2026-08-18 (pairs 0.888 against mismatches 0.237, 12/12 both directions
    in the mixed pool). No LLM call, but a 2.2 GB checkpoint load — named
    runs only. Deterministic model, so a change in this verdict is a change
    in the model cache, not noise."""
    if not _named_on_the_command_line(request):
        pytest.skip('loads the 2.2 GB encoder: run this file by name to opt in')
    pytest.importorskip('sentence_transformers')
    reply = widget.measure_bilingual_alignment.invoke({'model_name': ''})
    print('\n' + reply)
    assert 'Verdict: aligned' in reply
    # derived from the fixture, which is editable without touching Python
    pairs, _ = widget._read_pairs('')
    assert f'{len(pairs)}/{len(pairs)}' in reply
