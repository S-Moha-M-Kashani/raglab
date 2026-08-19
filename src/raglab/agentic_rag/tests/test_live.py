"""The agentic loop's live probe — collected everywhere, run only when named.

It pays for real OpenRouter calls, which the suite may never do silently — so
the guard is invocation-shaped, the same as tests/test_skills_live.py's: the
test runs only when this file (or a node id inside it) is named on the pytest
command line, and skips under any broad sweep (`pytest tests/`, a merge gate,
`uv run pytest`). It lives inside the package rather than in tests/ on
purpose: tests/conftest.py pins that suite offline and blanks the developer's
keys, and this file exists to do the one thing that plumbing forbids — run
the real loop against a real model with the real `.env`.

    uv run pytest src/raglab/agentic_rag/tests/test_live.py -v -s

tests/test_agentic_rag.py holds the offline coverage (caps, stop reasons,
graph shape, refusals — all against a scripted model); this file holds the
one claim only a real model can settle: that the compiled graph, prompted
with the real prompts, plans, retrieves, assesses, drafts and critiques its
way to a grounded answer, and that the row it produces says so.
"""
import json

import pytest

from raglab import agentic_rag, datasets
from raglab.config import LabConfig
from raglab.index import IndexRegistry
from raglab.llm import lab_llm
from raglab.models import Roles
from raglab.settings import load_lab_settings


def _named_on_the_command_line(request) -> bool:
    """Whether this run asked for this file by name, as the play button and
    an explicit terminal run do — a directory or bare sweep did not."""
    return any('test_live' in arg
               for arg in request.config.invocation_params.args)


def test_the_full_scope_answers_a_real_question_with_a_real_model(
        request, monkeypatch):
    # this is an end-to-end test
    """One question through `scope='full'` on the smoke corpus, against the
    real OpenRouter backend: the loop must finish without an error stop, hand
    back a non-refused answer, and its trace must show the graph actually
    walked — plan first, at least one retrieval hop, a draft. Asserts on the
    row's own account of itself (`agent_stop`, `agent_calls`, the trace)
    rather than on any one model's wording, and prints the answer and the
    diagnostics so the human who opted in can judge the part no assertion
    can."""
    if not _named_on_the_command_line(request):
        pytest.skip('real OpenRouter calls: run this file by name to opt in')
    monkeypatch.setenv('RAGLAB_LLM', 'openrouter')
    settings = load_lab_settings()
    if not settings.openrouter_api_key:
        pytest.skip('OPENROUTER_API_KEY is not in .env')

    corpus, ground_truth = datasets.load('smoke-mini')
    cfg = LabConfig.from_dict({
        'index': {'chunker': 'session', 'embedder': 'token-hash',
                  'contextual': False},
        'retrieval': {'k': 3, 'candidates': 12, 'rerank_depth': 6},
        'generation': {'answerer': 'llm'},
        'agent': {'scope': 'full', 'max_hops': 2, 'max_revisions': 1}})
    assert cfg.validate() == []

    index = IndexRegistry(settings, corpus).get(cfg.index)
    asked = ground_truth['questions'][0]
    trace: dict = {}
    outcome = agentic_rag.run(index, cfg, asked['question_fa'],
                              ground_truth['meta']['query_date'],
                              llm=lab_llm(settings), models=Roles(),
                              trace=trace)

    diagnostics = {key: value for key, value in outcome.diagnostics.items()
                   if key.startswith('agent')}
    print('\n--- the live run ---')
    print('question:', asked['question_fa'])
    print('answer:', outcome.answer)
    print('diagnostics:', json.dumps(diagnostics, ensure_ascii=False))
    print('nodes:', [visit['node'] for visit in trace['agent']])
    print('agent_ms:', outcome.timings.get('agent_ms'))
    print('--- end of the live run ---')

    # The loop finished on its own terms, not by dying.
    assert diagnostics['agent_stop'] != 'error', diagnostics
    assert 'agent_error' not in diagnostics, diagnostics
    assert diagnostics['agent_scope'] == 'full'
    assert diagnostics['agent_calls'] >= 2, 'the loop barely ran'
    # The graph actually walked: plan first, evidence fetched, an answer drafted.
    nodes = [visit['node'] for visit in trace['agent']]
    assert nodes[0] == 'plan'
    assert nodes.count('retrieve') >= 1
    assert 'draft' in nodes
    # And it answered rather than refusing an easy question over its own corpus.
    assert outcome.answer.strip()
    assert not outcome.abstained, outcome.answer
