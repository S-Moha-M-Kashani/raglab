from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from raglab.evaluation import run_evaluation as evaluate
from raglab.rag_components import question_to_answer_pipeline as pipeline
from raglab.corpora import diary_corpus_loader as corpus
from raglab.dashboard import inspector_server as inspector
from raglab.evaluation import deterministic_metrics as metrics
from raglab.dashboard import service_presentation as present
from raglab.evaluation.tests.archive_examples import completed_archive
from raglab.configuration.lab_config import (
    IndexConfig,
    LabConfig,
    RetrievalConfig,
    LabSettings)
from raglab.rag_components.indexing.index_builder_registry import IndexRegistry

from raglab.conftest import _finished, _font_size_literals, _radius_literals

LAB_SETTINGS = LabSettings(openrouter_api_key='', llm_provider='fake')
INSPECTOR_JS = inspector.STATIC / 'inspector.js'
INSPECTOR_HTML = inspector.STATIC / 'inspector.html'


def test_evidence_spans_and_mark_gold_agree_on_the_same_quote_either_direction():
    # this is a unit test
    """The green highlight (`evidence_spans`) and the gray/white gold split
    (`mark_gold`) are two views of one match, over the same shared
    normaliser — a chunk contained *by* a quote is gold with nothing verbatim
    to highlight, and an empty normalisation on either side must never mark
    everything gold."""
    quote = 'قسط‌بندی جریمه اوکی شد شیش قسط'
    text = f'خبر خوب: {quote}، از اول ماه دیگه'
    spans = present.evidence_spans(text, [quote])
    assert len(spans) == 1
    start, end = spans[0]
    assert text[start:end] == quote

    # gold by the *reverse* direction — the chunk sits inside the quote — so
    # there is nothing verbatim to mark, and no span may be guessed
    assert present.evidence_spans('شیش قسط', [quote]) == []
    # a quote that is simply absent
    assert present.evidence_spans('یک متن بی ربط', [quote]) == []
    # no quotes at all, and empty text
    assert present.evidence_spans(text, []) == []
    assert present.evidence_spans('', [quote]) == []

    # two quotes in one chunk come back in reading order, and touching or
    # overlapping ranges merge — two <mark>s over the same characters would
    # nest and render as a darker stripe nobody asked for
    a, b = 'اول ماه', 'ماه دیگه'
    merged = present.evidence_spans('قبل از اول ماه دیگه بود', [a, b])
    assert len(merged) == 1
    s, e = merged[0]
    assert 'قبل از اول ماه دیگه بود'[s:e] == 'اول ماه دیگه'

    quotes = [quote]
    texts = [
        'خبر خوب: قسط‌بندی جریمه اوکی شد شیش قسط، از اول ماه دیگه',  # contains quote
        'امروز هوا خیلی گرم بود و کاری پیش نرفت',                    # unrelated
    ]
    flags = inspector.mark_gold(texts, quotes)
    assert flags == [True, False]

    # quote longer than a small chunk: chunk contained by the quote also counts
    assert inspector.mark_gold(['شیش قسط'], quotes) == [True]
    # no quotes → nothing is gold
    assert inspector.mark_gold(texts, []) == [False, False]

    # empty normalisation (blank, whitespace-only, punctuation-only) → never gold
    assert inspector.mark_gold(['', '   ', '...!!!'], quotes) == [False, False, False]

    # a QUOTE that normalises to empty (punctuation-only, or a single short
    # token the tokeniser drops) must not mark every candidate gold
    assert inspector.mark_gold(
        ['یک متن کاملا بی ربط', 'هر چیز دیگر'], ['؟!...']) == [False, False]
    assert inspector.mark_gold(['یک متن کاملا بی ربط'], ['۶']) == [False]


# Real fixture, real chunker.
def test_a_question_reports_how_many_gold_chunks_existed_to_find():
    # this is an integration test
    """"1 gold" is not a result until you know it was 1 of how many. The
    denominator is how many chunks in the whole index hold this question's
    evidence, not how many evidence quotes the fixture lists — one quote can
    split across chunks and one chunk can carry two quotes."""
    gt = corpus.load_ground_truth()
    index = IndexRegistry(LAB_SETTINGS, corpus.load_diary()).get(
        IndexConfig(chunker='fixed-overlap', chunk_chars=500, overlap=100,
                    contextual=True, embedder='ascii-hash'))
    question = next(q for q in gt['questions'] if q.get('evidence'))
    quotes = [ev['quote'] for ev in question['evidence']]

    total = present.gold_available(index, quotes)
    # counted the same way a candidate is marked, over every chunk in the index
    assert total == sum(present.mark_gold([c.text for c in index.chunks], quotes))
    assert total >= 1, 'a question whose evidence is in the corpus must be findable'

    # and it travels on the row, beside the candidates it is the denominator for
    cfg = RetrievalConfig(retriever='hybrid-rrf', reranker='none', grader='none',
                          k=5, rerank_depth=20, time_filter=False,
                          multi_query=False)
    _outcome, trace = pipeline.retrieve_traced(
        index, cfg, question['question_fa'], gt['meta']['query_date'])
    row = evaluate.trace_row(question, trace, gold_present=total)
    assert row['gold_available'] == total
    found = sum(1 for c in row['trace']['candidates'] if c['gold'])
    assert found <= row['gold_available'], (
        'more gold retrieved than exists — the two are counted differently')

    # a caller that does not know the index still gets a row, with the count
    # absent rather than a wrong number standing in for it
    assert evaluate.trace_row(question, trace)['gold_available'] is None


# Real fixture, real chunker.
def test_a_traced_candidate_carries_spans_that_slice_back_to_the_quote():
    # this is an integration test
    """Every span on every candidate must slice out of that candidate's own
    text, and a candidate marked gold with a verbatim quote must carry at
    least one."""
    gt = corpus.load_ground_truth()
    index = IndexRegistry(LAB_SETTINGS, corpus.load_diary()).get(
        IndexConfig(chunker='fixed-overlap', chunk_chars=500, overlap=100,
                    contextual=True, embedder='ascii-hash'))
    cfg = RetrievalConfig(retriever='hybrid-rrf', reranker='none', grader='none',
                          k=5, rerank_depth=20, time_filter=False,
                          multi_query=False)
    question = next(q for q in gt['questions'] if q.get('evidence'))
    _outcome, trace = pipeline.retrieve_traced(
        index, cfg, question['question_fa'], gt['meta']['query_date'])
    row = evaluate.trace_row(question, trace)

    quotes = [ev['quote'] for ev in question['evidence']]
    verbatim_seen = 0
    for candidate in row['trace']['candidates']:
        spans = candidate['gold_spans']
        assert isinstance(spans, list)
        for start, end in spans:
            assert candidate['text'][start:end] in quotes
        if spans:
            verbatim_seen += 1
            assert candidate['gold'], 'a highlighted candidate must be gold'
        elif any(q in candidate['text'] for q in quotes):
            raise AssertionError('a verbatim quote was left unhighlighted')
    assert verbatim_seen, 'expected at least one candidate to highlight'


# Real in-memory index, offline ascii-hash embedder.
def test_retrieve_traced_records_ranks_and_dropped_candidates():
    # this is an integration test
    diary = corpus.load_diary()
    gt = corpus.load_ground_truth()
    index = IndexRegistry(LAB_SETTINGS, diary).get(
        IndexConfig(chunker='session', embedder='ascii-hash'))
    # rerank_depth(20) > k(3): mmr keeps 3, so at least 17 candidates are dropped.
    cfg = RetrievalConfig(retriever='hybrid-rrf', reranker='none',
                          grader='none', k=3, rerank_depth=20, time_filter=False)
    question = gt['questions'][0]['question_fa']
    query_date = gt['meta']['query_date']

    outcome, trace = pipeline.retrieve_traced(
        index, cfg, question, query_date)

    assert trace['candidates'], 'trace recorded no candidates'
    first = trace['candidates'][0]
    # every step is represented on each candidate row
    for key in ('dense_rank', 'bm25_rank', 'fused_rank',
                'retrieval_score', 'rerank_score', 'grade_score', 'kept'):
        assert key in first, f'missing {key}'
    # ranks are 1-based ints or None
    for cand in trace['candidates']:
        for key in ('dense_rank', 'bm25_rank', 'fused_rank'):
            assert cand[key] is None or (isinstance(cand[key], int) and cand[key] >= 1)
    # some candidate survived, some was dropped
    kept = [c for c in trace['candidates'] if c['kept']]
    dropped = [c for c in trace['candidates'] if not c['kept']]
    assert kept and dropped, 'expected both kept and dropped candidates'
    assert len(kept) == len(outcome.contexts)
    # the ordered step lists are present
    assert trace['dense'] and trace['bm25'] and trace['fused']

    # exercise the grader path: verify grade_score is populated as float when grader is active
    cfg_with_grader = RetrievalConfig(retriever='hybrid-rrf', reranker='none',
                                      grader='lexical', grade_threshold=0.0, k=3,
                                      rerank_depth=20, time_filter=False)
    outcome_graded, trace_graded = pipeline.retrieve_traced(
        index, cfg_with_grader, question, query_date)
    graded_candidates = [c for c in trace_graded['candidates']
                         if c['grade_score'] is not None]
    assert any(isinstance(c['grade_score'], float) for c in graded_candidates), \
        'expected at least one candidate with float grade_score'


def _client(monkeypatch):
    from raglab.dashboard import inspector_server as inspector
    # Pin the offline/fake backend so no test needs a key or a network.
    monkeypatch.setattr(inspector, 'load_lab_settings', lambda: LAB_SETTINGS)
    return TestClient(inspector.create_inspector_app())


def test_follow_advertises_only_the_active_archive_id(monkeypatch):
    # this is an integration test
    calls = []

    def lab_get(path):
        calls.append(path)
        if path == '/api/imported-archives/active':
            return {'archive_id': 'imported-run-001'}
        return {'jobs': []}

    monkeypatch.setattr(inspector, '_lab_get', lab_get)
    body = _client(monkeypatch).get('/api/follow').json()
    assert body['archive_id'] == 'imported-run-001'
    assert 'evaluation' not in body


def test_inspector_proxies_one_archive_and_return_to_live(monkeypatch):
    # this is an integration test
    full = completed_archive()
    monkeypatch.setattr(inspector, '_lab_get',
                        lambda path: full if path.endswith('imported-run-001') else None)
    deleted = []
    monkeypatch.setattr(inspector, '_lab_delete', lambda path: deleted.append(path) or {})
    client = _client(monkeypatch)
    assert client.get('/api/imported-archives/imported-run-001').json() == full
    assert client.delete('/api/imported-archives/active').status_code == 200
    assert deleted == ['/api/imported-archives/active']


def test_the_inspector_proxies_one_recorded_experiment(monkeypatch):
    # this is an integration test
    """The board's `↗` sends a reader here, so the Inspector needs to fetch a
    record. It proxies rather than reading raglab.db: this service owns no
    ledger, and giving it one would be a second writer to a record whose whole
    value is being written once."""
    record = {'experiment_id': 'exp-1', 'kind': 'run', 'dataset': 'smoke-mini',
              'detail': {'config': {}, 'rows': [], 'traces': []}}
    asked = []

    def lab_get(path):
        asked.append(path)
        return record if path == '/api/experiments/exp-1' else None

    monkeypatch.setattr(inspector, '_lab_get', lab_get)
    client = _client(monkeypatch)
    found = client.get('/api/experiments/exp-1')
    assert found.status_code == 200
    assert found.json()['experiment_id'] == 'exp-1'
    # Asked of the lab, over HTTP, and not of a database this process opened.
    assert asked == ['/api/experiments/exp-1']


def test_an_experiment_the_lab_cannot_produce_is_a_404_not_an_empty_view(
        monkeypatch):
    # this is an integration test
    """Better a stated 404 than a read-only view pinned to nothing, which reads
    as an experiment that recorded no evidence."""
    monkeypatch.setattr(inspector, '_lab_get', lambda path: None)
    client = _client(monkeypatch)
    assert client.get('/api/experiments/no-such-id').status_code == 404


# FastAPI TestClient over the read-only app.
def test_groundtruth_endpoint_returns_full_pairs(monkeypatch):
    # this is an integration test
    client = _client(monkeypatch)
    body = client.get('/api/groundtruth').json()
    q = body['questions'][0]
    # the fields the :9002 /api/questions endpoint strips must be present here
    for key in ('answer_fa', 'key_facts', 'evidence', 'question_fa',
                'type', 'difficulty', 'answerable'):
        assert key in q, f'missing {key}'
    assert 'quote' in q['evidence'][0]


@pytest.mark.parametrize('dataset, language', [
    ('', 'fa'), ('diary-fa', 'fa'), ('meetings-de', 'de'), ('support-en', 'en'),
])
def test_groundtruth_endpoint_names_the_corpus_language(monkeypatch, dataset,
                                                       language):
    # this is an integration test
    """The page cannot render a corpus in the direction it reads unless it is
    told which language that is, and it is not in `meta`: a ground-truth file's
    meta describes the question set, and `_split` writes no language into it for
    any corpus — the built-in diary keeps its own on the *corpus* half. So the
    route says it outright, resolved from the catalogue where the fact lives."""
    client = _client(monkeypatch)
    body = client.get(f'/api/groundtruth?dataset={dataset}').json()
    assert body['language'] == language, (
        'the Inspector rendered every corpus right-to-left in a Persian face '
        'because this fact never reached it'
    )


# FastAPI TestClient over the read-only app; real in-memory index build via
# the job runner. Parametrized over a flat build and a hierarchical one, both
# on the five-session smoke corpus, so the toggle's two halves (leaves-only,
# leaves-plus-summaries) are one test rather than two near-duplicates.
@pytest.mark.parametrize('index_cfg, expect_summaries', [
    ({'dataset': 'smoke-mini', 'chunker': 'session', 'embedder': 'token-hash'},
     False),
    ({'dataset': 'smoke-mini', 'chunker': 'session', 'embedder': 'token-hash',
      'hierarchy': 'kmeans', 'summarizer': 'centroid', 'min_group': 2},
     True),
], ids=['flat', 'hierarchy'])
def test_chunks_job_returns_sessions_and_any_summaries_beside_them(
        monkeypatch, tmp_path, smoke_index, index_cfg, expect_summaries):
    # this is an integration test
    """The manual build path serves both halves in one job, so the toggle
    needs no second request — and `total` keeps counting leaves, since that
    is what the chunk-size knob is read against. `kmeans` rather than
    `metadata` for the hierarchy half: the smoke corpus's five sessions each
    declare their own single topic, so a metadata grouping never reaches
    `min_group` and writes no summary at all — measured empirically before
    writing this test. This build is also the real behavioural half of the
    Inspector's no-ledger guard: `test_the_inspector_constructs_its_job_
    table_with_no_recorder` pins the source that keeps it that way, but only
    a real build asked to record and then checked can catch the Inspector
    reaching `ledger.record` by some other spelling that pin cannot name. A
    fresh `tmp_path` per parametrize case, never touched by anything else in
    the suite, rather than the session-wide `RAGLAB_DB` redirection every
    other test here shares — a file another test's real recording created
    there would make this assert fail for the wrong reason."""
    db = tmp_path / 'raglab.db'
    monkeypatch.setenv('RAGLAB_DB', str(db))
    client = _client(monkeypatch)
    acc = client.post('/api/chunks', json={'index': index_cfg})
    assert acc.status_code == 202
    job = _finished(client, acc.json()['job_id'])
    assert job['state'] == 'done', job.get('error')
    result = job['result']

    assert not db.exists(), (
        'the Inspector must not even create the ledger file for a scratch build')

    groups = result['chunks_by_session']
    assert result['total'] == sum(len(g['chunks']) for g in groups)
    first = groups[0]
    assert first['session_id'] and 'date' in first
    assert all('id' in c and 'text' in c for c in first['chunks'])
    # one group per session either way — a hierarchy adds rows beside the
    # leaves, it never folds two sessions' leaves into one group
    assert len(groups) == len(smoke_index.index.by_session)

    if expect_summaries:
        assert result['total_summaries'] == len(result['summaries']) >= 1
        summary = result['summaries'][0]
        assert summary['members'] >= 1 and summary['level'] >= 1 and summary['text']
    else:
        assert result['total_summaries'] == 0
        assert result['summaries'] == []


# FastAPI TestClient over the read-only app; real in-memory index build and
# retrieval trace via the job runner. The file's one job round trip left
# whole: build, retrieve, mark gold, all through the actual job table.
def test_trace_job_marks_gold(monkeypatch):
    # this is an integration test
    client = _client(monkeypatch)
    gt_q = client.get('/api/groundtruth').json()['questions'][0]
    payload = {'index': {'chunker': 'session', 'embedder': 'ascii-hash'},
               'retrieval': {'retriever': 'hybrid-rrf', 'reranker': 'none',
                             'grader': 'none', 'k': 3, 'rerank_depth': 20,
                             'time_filter': False},
               'question_id': gt_q['id']}
    acc = client.post('/api/trace', json=payload)
    assert acc.status_code == 202
    job = _finished(client, acc.json()['job_id'])
    assert job['state'] == 'done', job.get('error')
    cands = job['result']['trace']['candidates']
    assert cands and all('gold' in c for c in cands)

    # An invalid config must refuse synchronously, the same as /api/questions,
    # rather than accept the job and fail it with state='error' — the
    # Inspector's own route plumbing, which `test_query_rejects_an_unknown_
    # strategy` (test_server.py) does not exercise: that hits the lab's
    # `/api/queries` on :9002, a different service whose route calls `screen()`
    # before accepting a job for reasons CLAUDE.md records as having drifted
    # from this one before (the board's proxy tests, the panel/Inspector split).
    bad = client.post('/api/trace', json={
        'index': {'chunker': 'session', 'embedder': 'ascii-hash'},
        'retrieval': {'reranker': 'nope'}, 'question_id': gt_q['id']})
    assert bad.status_code == 400
    assert 'unknown reranker' in bad.json()['detail']


# --- the served Inspector page's own conventions, as one table -------------
#
# The six page-pin tests that used to live here (three-views, generation tab
# and evidence reveal, question picker, chunks/summaries toggle, JS keeps no
# config literal, the corpus-follow fixture wiring), plus the Inspector's own
# half of two tests that used to live in test_panel.py (the shared column
# sorter, the shared token sheet and script) are rows below.

@pytest.fixture(scope='module')
def inspector_texts():
    """Every named text the convention table checks, fetched the one way a
    browser actually reaches it (`client.get`) rather than a second disk read
    of the same file. A fresh client per fixture call, same as `_client`
    above, since the Inspector app is cheap to build and this keeps the
    fixture independent of any one test's monkeypatch."""
    from fastapi.testclient import TestClient

    client = TestClient(inspector.create_inspector_app())
    return {
        'inspector.html': client.get('/').text,
        'inspector.css': client.get('/inspector.css').text,
        'inspector.js': client.get('/inspector.js').text,
        # The shared chrome sheet, over its own route: the bar, the surface
        # switcher and — since the tables pass — the scroll region and sticky
        # header this page's own tables now sit inside.
        'chrome.css': client.get('/chrome.css').text,
        # The shared script, over its own route as well: the reveal placer both
        # surfaces need moved into it, so a claim about where a fixed reveal is
        # put is now a claim about this file rather than about inspector.js.
        'lab.js': client.get('/lab.js').text,
    }


# (file, must_contain, must_not_contain, reason) — one row per retired
# single-substring pin test, each carrying the one line that used to be its
# docstring so a failure names the rule rather than printing a bare
# "assert 'x' in text".
INSPECTOR_CONVENTIONS = [
    ('inspector.css', 'font-size: var(--t-sm)', None,
     'one table step for both surfaces: this table read --t-xs where the '
     "lab's read --t-sm, which is what a ramp offering three indiscriminable "
     'choices does to two pages written months apart'),
    ('inspector.css', None, '.retrieval-table { font-size: var(--t-2xs); }',
     'and no smaller type on a narrow screen — that step is under the '
     'readable floor, and what a narrow screen needs is a table that scrolls, '
     'which it now has at every width'),
    ('chrome.css', 'button.why::after', None,
     "the Inspector's '!' is not merely the same affordance as the lab's, it "
     'is the same rule: two copies in two units under comments on each side '
     'claiming they were one thing is what this replaces. It still clears the '
     'same 24×24 floor from a pseudo-element, for the same reason — a 24px '
     'disc at the end of a small label would set that row\'s line height'),
    ('inspector.css', None, '.inspector-why',
     'and the page keeps no copy of the mark: what stays here is only where '
     'the sentence it opens goes, which is genuinely page-local because these '
     'marks sit in a flex row of scores'),
    ('inspector.html', 'id="chunks-status" aria-live="polite"', None,
     'a build that finished, or failed, must say so — this span is the only '
     'place it is reported'),
    ('inspector.html', 'id="retrieval-status" aria-live="polite"', None,
     'the same for a question added by hand, which can take a while and can '
     'fail'),
    ('inspector.js', None, 'title="a summary this build',
     'that a row is a summary rather than the corpus\'s own words is the most '
     'important thing about it, and a tooltip publishes it to a mouse and to '
     'nothing else — it is a line in the reveal now, which opens to a '
     'keyboard as well'),
    ('inspector.html', 'role="tablist"', None,
     'the four views are a tablist: `aria-selected` on a plain <button> means '
     'nothing, so a screen reader was told which view was showing by nothing '
     'at all'),
    ('inspector.html', 'role="tabpanel"', None,
     'a tab must control a panel that says it is one, or `aria-controls` '
     'points at an anonymous section'),
    ('inspector.html', 'aria-controls="view-retrieval"', None,
     'each tab must name the panel it opens — checked on one real pairing, so '
     'a `role="tab"` added without the wiring cannot satisfy the rows above'),
    ('inspector.js', 'tab.tabIndex = on ? 0 : -1', None,
     'a tablist is one stop in the page tab order, not four; the roving '
     'tabindex is what makes the arrow keys the way between them rather than '
     'an addition nobody needs'),
    ('inspector.css', '.inspector-header-inner', None,
     'the header is a page-level band and centres on the shared measure: '
     'without its own inner band its title and tabs sat 77px left of every '
     'card on the page at any viewport wider than the measure'),
    ('inspector.js', None, 'query box on the lab',
     'the one-off-query view must not send the reader to a control that is '
     'not there: the lab has had no query box for a while, and an empty state '
     'naming one is worse than an empty state, because the reader goes '
     'looking'),
    ('inspector.js', 'POST /api/queries', None,
     'it must name what does still start a one-off query, so the empty state '
     'says something true rather than nothing'),
    ('inspector.html', 'tab-groundtruth', None,
     'the served shell must expose its ground-truth tab hook'),
    ('inspector.html', 'tab-chunks', None,
     'the served shell must expose its chunks tab hook'),
    ('inspector.html', 'tab-retrieval', None,
     'the served shell must expose its retrieval tab hook'),
    ('inspector.html', 'class="inspector-tab"', None,
     'the served shell must expose the tab-switching hook — checked as the '
     'quoted class value so the tab-strip wrapper (`class="inspector-tabs"`, '
     'plural) cannot satisfy it by prefix collision'),
    ('inspector.html', 'class="retrieval-table"', None,
     'the served shell must expose the retrieval table hook — checked as the '
     'quoted class value so the unrelated `<template '
     'id="retrieval-table-template">` wrapper cannot satisfy it by prefix '
     'collision'),
    ('inspector.html', 'inspector-active-config', None,
     "the followed view's config statement must be renderable"),
    ('inspector.html', 'inspector-answer', None,
     "the followed query's generated answer must be renderable"),
    ('inspector.html', 'retrieval-questions', None,
     'one table per question of the followed experiment must be renderable'),
    ('inspector.html', 'tab-generation', None,
     'the generation tab must expose its hook'),
    ('inspector.html', 'view-generation', None,
     'the generation view must expose its hook'),
    ('inspector.html', 'generation-questions', None,
     'the generation view must expose a hook to render its questions into'),
    ('inspector.html', 'generation-active-config', None,
     "the generation view's active-config statement must be renderable"),
    ('inspector.css', '.chunk-reveal', None,
     'the evidence reveal is CSS-driven and the class must exist'),
    ('inspector.css', '.evidence-mark', None,
     'the evidence highlight is CSS-driven and the class must exist'),
    ('inspector.css', '.retrieval-row:hover', None,
     'the reveal must open on hover, without a click'),
    ('inspector.html', 'add-question', None,
     'the question picker must expose the button that opens it'),
    ('inspector.html', 'id="question-picker"', None,
     'the question picker must expose its own hook — checked as the quoted '
     'id so its filter input and listbox (`id="question-picker-filter"`, '
     '`id="question-picker-list"`) cannot satisfy it by prefix collision'),
    ('inspector.html', 'question-picker-list', None,
     'the question picker must expose the listbox hook'),
    ('inspector.css', '.q-option--hard', None,
     'the picker must colour-code the hard questions'),
    ('inspector.css', '.q-option--medium', None,
     'the picker must colour-code the medium questions'),
    ('inspector.css', '.q-option--easy', None,
     'the picker must colour-code the easy questions'),
    ('inspector.css', '.q-option-detail', None,
     'the picker must expose the detail reveal class'),
    ('inspector.css', '.q-option:hover', None,
     'the picker detail must reveal on hover, without a click'),
    ('inspector.html', 'role="listbox"', None,
     'the picker must be reachable without a mouse'),
    ('inspector.html', 'aria-expanded', None,
     'the picker must state its open/closed state for assistive tech'),
    ('inspector.html', 'id="chunks-mode"', None,
     'the chunks/summaries toggle must expose its own hook — checked as the '
     'quoted id so its two buttons (`id="chunks-mode-chunks"`, '
     '`id="chunks-mode-summaries"`) cannot satisfy it by prefix collision'),
    ('inspector.html', 'chunks-mode-chunks', None,
     'the toggle must expose its chunks-half hook'),
    ('inspector.html', 'chunks-mode-summaries', None,
     'the toggle must expose its summaries-half hook'),
    ('inspector.html', 'summaries-body', None,
     'the summaries view must expose a hook to render rows into'),
    ('inspector.html', 'Summaries', None,
     'the tab must name summaries even when none were built, or a reader has '
     'no reason to think the index might hold them'),
    ('inspector.css', '.layer-badge', None,
     'the summaries view reuses the retrieval table\'s badge rather than '
     'inventing a second vocabulary for the same idea'),
    ('inspector.html', 'aria-pressed', None,
     'the pressed state is what says which half of the toggle is on screen'),
    # Asserted on the *values*, not on a literal's name, so renaming a
    # hand-maintained copy of the pipeline cannot smuggle it back in.
    # `hybrid-rrf` is deliberately not a row here — it also appears in a
    # comment describing the active-config line, and a guard that forbids
    # describing the code is a guard nobody keeps.
    ('inspector.js', None, 'semantic-drift',
     "inspector.js must not keep its own copy of the pipeline's chunker name"),
    ('inspector.js', None, 'sentence-transformers',
     "inspector.js must not keep its own copy of the pipeline's embedder name"),
    ('inspector.js', None, 'grade_threshold',
     "inspector.js must not keep its own copy of the pipeline's gate threshold"),
    ('inspector.js', "fetch('/api/config')", None,
     'inspector.js must fetch the config it renders, rather than assume one — '
     "checked against the actual `fetch(...)` call rather than the bare route, "
     'which also appears in an adjacent comment describing why `CHOSEN` is a '
     'fallback'),
    ('inspector.js', 'FOLLOWED_CONFIG || CHOSEN', None,
     'following the lab must stay the primary path and the served config only '
     'the fallback for a lab that is down — checked against the actual '
     'fallback expression rather than the bare identifier, which also appears '
     'in a comment naming it three lines above its declaration'),
    ('inspector.js', "'/api/groundtruth?dataset='", None,
     'the ground-truth fixture must be fetched with the corpus named, or a '
     'dataset change has nothing to reload against'),
    ('inspector.js', "(body.dataset || '') !== followed.dataset", None,
     'the follow loop must notice the corpus changing'),
    ('inspector.js', 'followDataset(followed.dataset)', None,
     'a changed corpus must reload the fixture'),
    ('inspector.js', 'await loadGroundTruth(dataset)', None,
     'the reload must name the corpus it was given'),
    ('inspector.js', 'dataset: FOLLOWED_DATASET', None,
     'a question added from the picker must be run against the corpus the '
     'picker offered, not a stale default'),
    ('inspector.html', 'sorttable.js', None,
     'the Inspector must load the shared column sorter, the same one the '
     'panel loads'),
    ('inspector.html', 'data-nosort', None,
     "the ranks column draws a shape the same three numbers already follow, "
     "so sorting on the picture would sort on nothing"),
]


@pytest.mark.parametrize('file, must_contain, must_not_contain, reason',
                         INSPECTOR_CONVENTIONS)
def test_the_served_inspector_page_keeps_its_conventions(
        inspector_texts, file, must_contain, must_not_contain, reason):
    # this is a convention test
    """Six page-pin tests plus two halves moved over from test_panel.py,
    folded into one table. Each row is a claim the served Inspector shell
    makes about itself, and the reason string is what a failure prints
    instead of a bare `assert 'x' in text`."""
    text = inspector_texts[file]
    if must_contain is not None:
        assert must_contain in text, reason
    if must_not_contain is not None:
        assert must_not_contain not in text, reason


def test_inspector_archive_mode_reuses_renderers_and_fetches_once_per_id():
    # this is a convention test
    source = INSPECTOR_JS.read_text()
    function = source[source.index('async function followImportedArchive'):
                      source.index('function renderFollow')]
    assert 'if (archiveId === activeArchiveId) return;' in function
    assert 'const archive = await archiveRequest(' in function
    assert "'/api/imported-archives/' + encodeURIComponent(archiveId)" in function
    render = source[source.index('function renderImportedArchive'):
                    source.index('async function followImportedArchive')]
    for call in ('renderGroundTruth(', 'renderChunkGroups(',
                 'renderQuestionTables(', 'renderGeneration('):
        assert call in render
    html = INSPECTOR_HTML.read_text()
    assert 'id="archive-state"' in html
    assert 'id="archive-return-live"' in html


def test_archive_mode_still_reports_lab_reachability():
    # this is a convention test
    """A page opened while an archive is already active must not sit on the
    boot placeholder ("looking for the lab…") forever: the archive branch of
    `renderFollow` returns early, so the follow state has to be written before
    that branch, from the same body that carried the archive id."""
    source = INSPECTOR_JS.read_text()
    follow = source[source.index('async function renderFollow'):]
    assert follow.index('setFollowState(body);') \
        < follow.index('if (body.archive_id)'), (
        'renderFollow must set the follow state before the early-returning '
        'archive branch, or archive mode never reports lab reachability')


def test_the_inspector_reads_a_corpus_in_its_own_direction(inspector_texts):
    # this is a convention test
    """The largest honesty gap this page had: it rendered every corpus
    right-to-left in a Persian face because the first one was a Farsi diary.
    Four of the five bundled corpora are German or English, and all four came
    out reversed, in Vazirmatn, with the chunk column against the wrong edge of
    its own column. Nothing was broken — the page simply never asked, while
    `/api/groundtruth` knew.

    So: no literal `rtl` anywhere in the page's markup or its script, the
    Persian face reachable only through a direction, and the chunk column's edge
    read from the one attribute the page sets from the reported language."""
    js, css, html = (inspector_texts['inspector.js'],
                     inspector_texts['inspector.css'],
                     inspector_texts['inspector.html'])
    assert 'dir="rtl"' not in js and 'dir="rtl"' not in html, (
        'a hardcoded direction is a claim about a corpus the page has not been '
        'told the language of — there were fourteen of these in the script and '
        'one in the markup, and the markup is fixed at page load while the '
        'corpus is not'
    )
    assert 'setCorpusDir(body.language)' in js, (
        'the direction comes from the language the route reports, resolved '
        'before the first row is written'
    )
    # Every rule that names the Persian stack must sit behind a direction. The
    # declaration and the token's own definition are the two that may not.
    for line in css.splitlines():
        if 'var(--farsi)' not in line or '--farsi:' in line:
            continue
        assert '[dir="rtl"]' in line, (
            f'this rule pins the Persian face with nothing guarding it: {line.strip()!r} '
            '— it was on the chunk preview, the reveal and the answer box, so '
            'an English corpus was set in Vazirmatn'
        )
    assert ':root[data-corpus-dir="rtl"] .retrieval-table td.chunk-cell' in css, (
        "the chunk column's edge follows the corpus, and where a column sits "
        'is layout — so it reads the direction from the page root rather than '
        'from the text inside it'
    )


def test_the_inspector_shares_one_token_sheet_and_one_script_with_the_panel():
    # this is a convention test
    """`tokens.css`, `chrome.css`, `lab.js` and `sorttable.js` are one file
    for both pages rather than a copy each, so a design token, the top bar, a
    utility or the meaning of clicking a column header cannot drift apart on
    either page. This pins that the Inspector actually routes each of them, its
    page actually loads them, and each loads before the page's own stylesheet
    or script — a later link would lose the shared rules to the page's own
    overrides instead of feeding them. The panel's half of this claim lives in
    test_panel.py."""
    from fastapi.testclient import TestClient

    client = TestClient(inspector.create_inspector_app())
    shared_css = ('tokens.css', 'chrome.css')
    shared_js = ('lab.js', 'sorttable.js')
    for name in shared_css + shared_js:
        assert (inspector.STATIC / name).exists(), name

    html = client.get('/').text
    for name in shared_css:
        served = client.get(f'/{name}')
        assert served.status_code == 200, name
        assert served.headers['content-type'].startswith('text/css'), name
        assert html.index(f'href="/{name}"') < html.index('href="/inspector.css"'), (
            f'{name} must be linked before the page\'s own sheet, or the page '
            'overrides the shared rules instead of building on them')
    for name in shared_js:
        served = client.get(f'/{name}')
        assert served.status_code == 200, name
        assert served.headers['content-type'].startswith(
            'application/javascript'), name
        assert html.index(f'src="/{name}"') < html.index('src="/inspector.js"'), (
            f'{name} must load before inspector.js, which calls into it')


def test_the_inspector_draws_its_scale_from_the_shared_sheet(inspector_texts):
    # this is a convention test
    """The Inspector carried 16 hand-set type sizes and its own radii, and
    agreed with the panel on none of them. Both surfaces read one scale or the
    shared sheet is decoration. The panel's half of this claim lives in
    test_panel.py."""
    css = inspector_texts['inspector.css']
    assert _font_size_literals(css) == []
    assert _radius_literals(css) == []


def test_the_inspector_centres_on_the_shared_measure(inspector_texts):
    # this is a convention test
    """The Inspector set its own 92rem measure, its own 1.25rem gutter, and
    never centred itself, so the two surfaces disagreed about how wide a page
    is and where its left edge falls."""
    css = inspector_texts['inspector.css']
    assert '92rem' not in css, 'the Inspector must not keep its own measure'
    assert 'max-width: var(--measure)' in css
    assert 'margin: 0 auto' in css, (
        'the Inspector was left-aligned while the panel was centred')


def test_the_inspector_tables_sit_in_the_shared_scroll_region(inspector_texts):
    # this is a convention test
    """The retrieval table is nine columns, and until the shared region reached
    this page its wrapper scrolled only under 46rem — because at any wider width
    it clipped the chunk reveal hanging below a row. The region is real at every
    width now and the reveal is fixed to the viewport instead of trapped inside
    it, which is what makes both possible at once: a table a keyboard can scroll
    across, and a chunk you can read."""
    js = inspector_texts['inspector.js']
    css = inspector_texts['inspector.css']
    chrome = inspector_texts['chrome.css']
    assert "box.className = 'table-scroll'" in js
    assert "box.setAttribute('role', 'region')" in js, (
        'an overflow div that cannot take focus cannot be scrolled by a '
        'keyboard at all — arrows, PageUp/PageDown, Home and End all do '
        'nothing, so the mouse is the only way across nine columns')
    assert 'position: fixed' in css.split('.chunk-reveal {')[1].split('}')[0], (
        'the reveal must leave the scroll region\'s flow, or the region clips '
        'the text it exists to show')
    assert 'function placeReveal' in inspector_texts['lab.js'], (
        'a fixed reveal has to be told where to go, and told at the moment it '
        'opens — its row may have been scrolled anywhere inside the region. It '
        'lives in the shared script because the board on :9002 hangs a reveal '
        'off a cell in this same region and needs the same answer'
    )
    assert "placeReveal(revealCell(event.target), '.chunk-reveal')" in js, (
        'and this page must actually call it — only the caller knows which '
        'reveal its cell holds')
    assert '.table-scroll { overflow-x: auto; }' not in css, (
        'the narrow-width-only scroll override is what the shared region '
        'replaced; keeping both means the page has two answers')
    assert '.table-scroll thead th' in chrome, (
        'the sticky header belongs to the region, not to `.data-table`: this '
        "page's retrieval table keeps its own centred columns and its own "
        'row backgrounds, which carry the gold verdict and so cannot also be '
        'a stripe')


def test_the_agent_ladder_is_wired_to_the_shared_sorter(inspector_texts):
    # this is a convention test
    """The ladder was the one table on either surface built by a path that
    never reached `SortTable.make`. Both questions a reader brings to a loop
    trace are column questions — sort by node and a node visited three times
    collects itself; sort by hop and you see what each hop cost — and the third
    click puts back the order it was served in, which for this table is the
    sequence itself."""
    js = inspector_texts['inspector.js']
    ladder = js[js.index('function agentLadder'):js.index('function questionBlock')]
    assert 'SortTable.make(' in ladder


# --- following the lab (:9002) ----------------------------------------------
#
# `/api/follow` has two halves: a real HTTP round trip to :9002 (`_lab_get`),
# and a set of pure selection/normalisation functions that pick the newest
# relevant job and reshape it for the page. The six tests that used to drive
# all of this through a threaded fake HTTP server are now one integration
# test proving the transport (a real socket, both up and down) plus direct
# unit tests on each normalisation function — fed canned job lists, no
# thread, no socket, and a monkeypatched `_lab_get` standing in for the one
# network call those functions still make to fetch a job's full body.

import json as _json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FAKE_INDEX_JOB = {
    'id': 'idx-fake-1', 'kind': 'index', 'state': 'done',
    'config': {'index': {'chunker': 'session', 'embedder': 'ascii-hash'}},
    'result': {'chunks': 1, 'chunks_by_session': [
        {'session_id': 's1', 'date': '2026-01-01',
         'chunks': [{'id': 's1-0', 'text': 'chunk one'}]}]}}

FAKE_QUERY_JOB = {
    'id': 'q-fake-1', 'kind': 'query', 'state': 'done',
    'config': {'retrieval': {'retriever': 'hybrid-rrf', 'k': 8}},
    'result': {
        'question': 'یک سوال؟', 'question_id': 'q-001', 'answer': 'یک جواب.',
        'trace': {'candidates': [{'chunk_id': 's1-0', 'text': 'chunk one',
                                  'gold': True, 'dense_rank': 1, 'bm25_rank': 1,
                                  'fused_rank': 1, 'kept': True}]}}}


FAKE_CANDIDATE = {'chunk_id': 's1-0', 'text': 'chunk one', 'gold': True,
                  'dense_rank': 2, 'bm25_rank': 1, 'fused_rank': 1,
                  'rerank_score': 0.71, 'grade_score': None, 'kept': True}

# A retrieval-only run over the two questions an experiment selected.
FAKE_RETRIEVE_JOB = {
    'id': 'ret-fake-1', 'kind': 'retrieve', 'state': 'done',
    'config': {'retrieval': {'retriever': 'hybrid-rrf', 'k': 3}},
    'result': {'selection': {'n': 2},
               'questions': [
                   {'question_id': 'q-001', 'question_fa': 'سوال یک؟',
                    'trace': {'candidates': [FAKE_CANDIDATE]}},
                   {'question_id': 'q-002', 'question_fa': 'سوال دو؟',
                    'trace': {'candidates': [FAKE_CANDIDATE]}}]}}

# A judged evaluation, which carries the same per-question traces under its own
# key — the eval path scores as well as retrieves, so its rows live elsewhere.
FAKE_RUN_JOB = {
    'id': 'run-fake-1', 'kind': 'run', 'state': 'done',
    'config': {'index': {'chunker': 'semantic-drift',
                         'embedder': 'sentence-transformers'},
               'retrieval': {'retriever': 'dense', 'k': 8}},
    'result': {'run_id': '20260804-000000-abcdef',
               # a row as `metrics.score_question` builds it: the generated
               # answer plus the deterministic generation scores
               'rows': [{'id': 'q-009', 'type': 'single-hop', 'answerable': True,
                         'answer': 'یک جواب که مدل نوشت.',
                         'answer_similarity': 0.42, 'key_fact_coverage': 0.5,
                         'abstained': False, 'recall': 1.0}],
               'summary': {'n_questions': 1, 'overall': {'answer_similarity': 0.42}},
               'ragas': {'metrics': {'faithfulness': 0.75}, 'decision': 0.7},
               'traces': [{'question_id': 'q-009', 'question_fa': 'سوال نه؟',
                           'trace': {'candidates': [FAKE_CANDIDATE]}}],
               # An evaluation builds its index implicitly, so it reports the
               # chunks it used itself — there is no index job to read them from.
               'chunks_by_session': [
                   {'session_id': 'run-s1', 'date': '2026-02-02',
                    'chunks': [{'id': 'run-s1-0', 'text': 'drifted chunk'}]}]}}

# A build on a corpus that is not the built-in diary — the shape of the fault
# `_followed_dataset` exists to pin: the lab names its dataset on every job it
# starts, and the Inspector must read that rather than guess the diary.
FAKE_OTHER_CORPUS_JOB = {
    'id': 'idx-fake-2', 'kind': 'index', 'state': 'done',
    'config': {'index': {'chunker': 'session', 'embedder': 'ascii-hash',
                         'dataset': 'meetings-de'}},
    'result': {'chunks': 1, 'chunks_by_session': [
        {'session_id': 'mtg-0113', 'date': '2026-01-13',
         'chunks': [{'id': 'mtg-0113:c0', 'text': 'Protokoll'}]}]}}

# Newest first, the order the lab's own /api/jobs uses. The transport test
# reassigns this to say which run happened last.
FAKE_ORDER = [FAKE_QUERY_JOB, FAKE_INDEX_JOB]


class _FakeLabHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # a canned test server has nothing worth logging

    def do_GET(self):
        by_id = {job['id']: job for job in FAKE_ORDER}
        if self.path == '/api/jobs':
            body = {'jobs': [{'id': job['id'], 'kind': job['kind'],
                              'state': job['state'], 'config': job['config']}
                             for job in FAKE_ORDER]}
        elif self.path.startswith('/api/jobs/') and \
                self.path.split('/')[-1] in by_id:
            body = by_id[self.path.split('/')[-1]]
        else:
            self.send_response(404)
            self.end_headers()
            return
        payload = _json.dumps(body).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def fake_lab():
    """A tiny stand-in :9002 — canned `/api/jobs` and `/api/jobs/{id}` JSON —
    so the one transport test that needs a real socket stays fast, offline
    and independent of the real lab's own behaviour."""
    server = ThreadingHTTPServer(('127.0.0.1', 0), _FakeLabHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}'
    finally:
        server.shutdown()
        thread.join(timeout=2)


def _index_of(jobs: list[dict]) -> dict:
    """The `{'jobs': [...]}` shape `/api/jobs` returns — the summary list the
    pure functions below take as their one argument."""
    return {'jobs': [{'id': j['id'], 'kind': j['kind'], 'state': j['state'],
                      'config': j['config']} for j in jobs]}


def _canned_lab(jobs: list[dict]):
    """A monkeypatch stand-in for `inspector._lab_get`: a dict lookup by job
    id over canned full job bodies, so `_job_view`/`_question_set`/
    `_newest_chunks` — which each fetch one job's full body once they have
    picked it from the summary list — need no thread and no socket either."""
    by_id = {job['id']: job for job in jobs}
    def _get(path):
        return by_id.get(path.rsplit('/', 1)[-1])
    return _get


# FastAPI TestClient; the lab is a real thread over a real socket — the one
# thing worth proving with one, since everything the transport carries is
# proven directly below.
def test_follow_reports_the_lab_transport_up_and_down(monkeypatch, fake_lab,
                                                     request):
    # this is an integration test
    """`GET /api/follow` must answer HTTP 200 whether or not :9002 can be
    reached — an unreachable daemon is a normal state for the Inspector, not
    an exception, and names no corpus rather than the diary — and when it
    can, every key the page's `fetch('/api/follow')` reads must actually be
    on the response: not just `lab`/`index`/`query` but `dataset` and the
    set-wide `retrieval`/`generation` views, exactly what the reachable lab's
    newest finished jobs said."""
    monkeypatch.setenv('RAGLAB_INSPECTOR_LAB_URL', 'http://127.0.0.1:9')
    client = _client(monkeypatch)
    res = client.get('/api/follow')
    assert res.status_code == 200
    body = res.json()
    assert body['lab'] == 'down'
    assert body['index'] is None and body['query'] is None
    assert body['dataset'] == ''

    monkeypatch.setenv('RAGLAB_INSPECTOR_LAB_URL', fake_lab)
    client = _client(monkeypatch)
    body = client.get('/api/follow').json()
    assert body['lab'] == 'up'
    assert body['index']['config']['index']['chunker'] == 'session'
    assert body['index']['chunks_by_session'][0]['session_id'] == 's1'
    assert body['query']['config']['retrieval']['retriever'] == 'hybrid-rrf'
    assert body['query']['answer'] == 'یک جواب.'
    assert body['query']['question_id'] == 'q-001'
    assert body['query']['trace']['candidates'][0]['gold'] is True

    # A run naming a corpus, reassigned so the lab-up block also proves
    # `dataset` and the set-wide `retrieval`/`generation` views travel —
    # not just `lab`/`index`/`query`, which is all the pre-review version of
    # this test checked.
    module = request.module
    run_job_with_dataset = {
        **FAKE_RUN_JOB,
        'config': {**FAKE_RUN_JOB['config'],
                  'index': {**FAKE_RUN_JOB['config']['index'],
                            'dataset': 'smoke-mini'}}}
    monkeypatch.setattr(module, 'FAKE_ORDER',
                        [run_job_with_dataset, FAKE_INDEX_JOB])
    body = client.get('/api/follow').json()
    assert body['dataset'] == 'smoke-mini'
    assert body['retrieval']['kind'] == 'run'
    assert body['generation']['job_id'] == FAKE_RUN_JOB['id']


def test_question_set_prefers_the_newest_finished_set_over_an_older_one(monkeypatch):
    # this is a unit test
    """The retrieval window must show *only* the questions the experiment
    picked, and whichever of the two set-wide kinds finished more recently —
    both feed `_question_set`, which normalises them to one shape so the
    page keeps one renderer. Fed canned job lists directly; no thread, no
    socket."""
    monkeypatch.setattr(inspector, '_lab_get',
                        _canned_lab([FAKE_RETRIEVE_JOB, FAKE_RUN_JOB, FAKE_INDEX_JOB]))
    view = inspector._question_set(
        _index_of([FAKE_RETRIEVE_JOB, FAKE_RUN_JOB, FAKE_INDEX_JOB]))
    assert view['kind'] == 'retrieve'
    assert view['config']['retrieval']['k'] == 3
    assert [q['question_id'] for q in view['questions']] == ['q-001', 'q-002']
    candidate = view['questions'][0]['trace']['candidates'][0]
    assert candidate['gold'] is True and candidate['fused_rank'] == 1

    # an evaluation that finished later is what the window follows instead, and
    # its traces arrive under a different key on the lab's side
    monkeypatch.setattr(inspector, '_lab_get',
                        _canned_lab([FAKE_RUN_JOB, FAKE_RETRIEVE_JOB, FAKE_INDEX_JOB]))
    view = inspector._question_set(
        _index_of([FAKE_RUN_JOB, FAKE_RETRIEVE_JOB, FAKE_INDEX_JOB]))
    assert view['kind'] == 'run'
    assert [q['question_id'] for q in view['questions']] == ['q-009']

    # no set-wide run at all is a normal state, not an error
    assert inspector._question_set(_index_of([FAKE_INDEX_JOB])) is None


def test_newest_chunks_follows_whichever_job_reported_them(monkeypatch):
    # this is a unit test
    """The chunks window must describe the same pipeline the retrieval window
    does: not `kind == 'index'` but "the newest job that reported any chunks
    at all", since an evaluation builds its index implicitly and creates no
    index job of its own. And a job recorded before the lab reported summaries
    at all must come back with `summaries == []`, never a missing key."""
    # the exact shape that misled: the run is newer than the index build, and
    # they name different chunkers
    monkeypatch.setattr(inspector, '_lab_get',
                        _canned_lab([FAKE_RUN_JOB, FAKE_INDEX_JOB, FAKE_RETRIEVE_JOB]))
    index_view = inspector._newest_chunks(_index_of([FAKE_RUN_JOB, FAKE_INDEX_JOB]))
    assert index_view['config']['index']['chunker'] == 'semantic-drift'
    assert index_view['chunks_by_session'][0]['session_id'] == 'run-s1'
    assert index_view['summaries'] == []
    # and both windows now agree about which index produced what is on screen
    retrieval_view = inspector._question_set(
        _index_of([FAKE_RUN_JOB, FAKE_RETRIEVE_JOB, FAKE_INDEX_JOB]))
    assert (index_view['config']['index']['chunker']
            == retrieval_view['config']['index']['chunker'])

    # an explicit build afterwards is the newest again, and wins
    monkeypatch.setattr(inspector, '_lab_get',
                        _canned_lab([FAKE_INDEX_JOB, FAKE_RUN_JOB]))
    index_view = inspector._newest_chunks(_index_of([FAKE_INDEX_JOB, FAKE_RUN_JOB]))
    assert index_view['config']['index']['chunker'] == 'session'
    assert index_view['summaries'] == []

    # a hierarchy actually reports its summaries beside the leaves
    with_summaries = {
        'id': 'idx-fake-3', 'kind': 'index', 'state': 'done',
        'config': {'index': {'chunker': 'session', 'hierarchy': 'metadata'}},
        'result': {'chunks': 2, 'chunks_by_session': [
            {'session_id': 's1', 'date': '2026-01-01',
             'chunks': [{'id': 's1-0', 'text': 'chunk one'}]}],
            'summaries': [{'id': 'summary:h1-000', 'text': 'a group card',
                           'group_id': 'h1-000', 'level': 1, 'members': 2,
                           'member_ids': ['s1-0', 's2-0'], 'sessions': 2,
                           'chars': 12}]}}
    monkeypatch.setattr(inspector, '_lab_get', _canned_lab([with_summaries]))
    view = inspector._newest_chunks(_index_of([with_summaries]))
    assert view['chunks_by_session'][0]['session_id'] == 's1'
    assert len(view['summaries']) == 1
    assert view['summaries'][0]['group_id'] == 'h1-000'
    assert view['summaries'][0]['sessions'] == 2


def test_generation_view_only_when_an_evaluation_wrote_rows(monkeypatch):
    # this is a unit test
    """What the model wrote and how it scored come from the evaluation's own
    rows; only an evaluation has them, so a retrieval-only run leaves
    `_generation_view` returning `None` rather than showing a stale answer
    beside fresh ranks."""
    monkeypatch.setattr(inspector, '_lab_get',
                        _canned_lab([FAKE_RUN_JOB, FAKE_INDEX_JOB]))
    view = inspector._generation_view(_index_of([FAKE_RUN_JOB, FAKE_INDEX_JOB]))
    assert view['job_id'] == FAKE_RUN_JOB['id']
    assert view['config']['retrieval']['retriever'] == 'dense'
    row = view['rows'][0]
    assert row['id'] == 'q-009'
    assert row['answer'] == 'یک جواب که مدل نوشت.'
    assert row['answer_similarity'] == 0.42
    # the run-level judged scores, which are per run and not per question
    assert view['ragas']['metrics']['faithfulness'] == 0.75
    assert view['summary']['n_questions'] == 1

    # a retrieval-only run generates nothing, and says so rather than lying
    monkeypatch.setattr(inspector, '_lab_get',
                        _canned_lab([FAKE_RETRIEVE_JOB, FAKE_INDEX_JOB]))
    assert inspector._generation_view(
        _index_of([FAKE_RETRIEVE_JOB, FAKE_INDEX_JOB])) is None


def test_followed_dataset_reads_the_newest_jobs_own_config():
    # this is a unit test
    """Which corpus the lab is on is a fact about the whole page, so
    `_followed_dataset` answers it once from the same summary list rather
    than each window guessing separately — and it needs no `_lab_get` call at
    all, since the dataset already rides on `/api/jobs`'s own config field."""
    assert inspector._followed_dataset(
        _index_of([FAKE_OTHER_CORPUS_JOB, FAKE_INDEX_JOB])) == 'meetings-de'
    # the newest job wins, exactly as every other window on this page follows
    # the newest job — switching back to the diary must switch the fixture back
    assert inspector._followed_dataset(
        _index_of([FAKE_INDEX_JOB, FAKE_OTHER_CORPUS_JOB])) == ''
    # a job whose config names no index at all cannot say which corpus it ran
    # on, so it is passed over rather than read as the built-in one: "does not
    # say" and "says the diary" are different facts
    assert inspector._followed_dataset(
        _index_of([FAKE_QUERY_JOB, FAKE_OTHER_CORPUS_JOB])) == 'meetings-de'
    assert inspector._followed_dataset({'jobs': []}) == ''


# FastAPI TestClient over the read-only app; a real in-memory index build and
# one job round trip through the actual `/api/questions` route, compared
# against a direct, untraced `pipeline.retrieve` call — two genuinely
# different code paths, so the comparison below is not one call checked
# against itself.
def test_adding_a_question_produces_rows_identical_to_the_run_s_own(monkeypatch):
    # this is an integration test
    """A question you add by hand has to arrive scored exactly like the ones
    the experiment selected — same generation row, same metric keys — or the
    two cannot be read side by side. Compared against the actual `/api/
    questions` route's own row (one real job round trip), not a second
    hand-written copy of its logic: an earlier version of this test computed
    *both* sides itself by calling `retrieve_traced`/`answer`/`score_question`
    directly, so the route's own generation branch — `run_question`'s work()
    closure in inspector_server.py — was touched by nothing but the 404 check below,
    and a row measured under the wrong `k` or missing its `pipeline.answer`
    call would still have passed."""
    gt = corpus.load_ground_truth()
    gt_q = gt['questions'][0]
    query_date = gt['meta']['query_date']
    config = {'index': {'chunker': 'fixed-overlap', 'chunk_chars': 500,
                        'overlap': 100, 'contextual': True,
                        'embedder': 'ascii-hash'},
              'retrieval': {'retriever': 'hybrid-rrf', 'reranker': 'lexical',
                            'grader': 'none', 'k': 5, 'rerank_depth': 20,
                            'time_filter': False, 'multi_query': False},
              'generation': {'answerer': 'extractive'}}
    cfg = LabConfig.from_dict(config)
    index = IndexRegistry(LAB_SETTINGS, corpus.load_diary()).get(cfg.index)

    # the run's own path: every evaluation retrieves with the plain,
    # untraced call, computed directly — the reference the route's own row
    # below is compared against
    outcome_b = pipeline.retrieve(index, cfg.retrieval, gt_q['question_fa'],
                                  query_date)
    outcome_b = pipeline.answer(outcome_b, cfg.generation)
    reference = metrics.score_question(gt_q, outcome_b, cfg.retrieval.k)

    # the added-question path: the real route, over HTTP, through the job
    # runner
    client = _client(monkeypatch)
    acc = client.post('/api/questions', json={**config, 'question_id': gt_q['id']})
    assert acc.status_code == 202, acc.text
    job = _finished(client, acc.json()['job_id'])
    assert job['state'] == 'done', job.get('error')
    result = job['result']

    # the route's whole response shape — a rename or a dropped key in
    # `run_question` would break the page's Add-question flow and, before
    # this assert, no test would say so
    assert set(result) >= {'config', 'retrieval', 'generation'}
    # and it says which config produced it, because a row measured under other
    # settings than its neighbours is worse than no row
    assert result['config']['index']['chunker'] == 'fixed-overlap'

    # the retrieval half, shaped like a followed question
    retrieval = result['retrieval']
    assert retrieval['question_id'] == gt_q['id']
    assert isinstance(retrieval['gold_available'], int)
    candidate = retrieval['trace']['candidates'][0]
    for key in ('dense_rank', 'bm25_rank', 'fused_rank', 'rerank_score',
                'grade_score', 'kept', 'gold', 'gold_spans'):
        assert key in candidate, f'missing {key}'

    # the generation half, shaped like an evaluation's row: same keys, so the
    # added question shows the same metrics and no others — the route's own
    # row against the direct reference, so a regression in either one shows
    row = result['generation']
    assert set(row) == set(reference), (
        f"added row differs: only here {set(row) - set(reference)}, "
        f"only in the eval row {set(reference) - set(row)}")
    assert row['id'] == gt_q['id'] and row['answer']

    # an unknown id refuses synchronously rather than dying inside a job
    assert client.post('/api/questions',
                       json={**config, 'question_id': 'q-nope'}).status_code == 404


# FastAPI TestClient over the read-only app.
def test_explain_serves_the_same_metric_help_the_lab_does(monkeypatch):
    # this is an integration test
    """Served from `explain` — the lab's own source for /api/options —
    rather than copied into the Inspector's page, so the two panels cannot
    end up explaining the same metric differently."""
    from raglab.configuration import explainer_assembly as explain
    client = _client(monkeypatch)
    body = client.get('/api/explain').json()

    assert body['metrics'] == explain.measures()
    assert body['help'] == explain.topics()
    # the generation half specifically, since that is what the new tab grades
    generation = {m['key'] for m in body['metrics'] if m['step'] == 'generation'}
    assert {'answer_similarity', 'key_fact_coverage', 'faithfulness',
            'abstained_correctly'} <= generation
    # every measure carries the text the '!' opens, or the mark has nothing to say
    assert all(m.get('formula') or m.get('note') or body['help'].get(f"metric.{m['key']}")
               for m in body['metrics'])


# --- summaries: the rows a hierarchy adds beside the leaves -----------------

HIERARCHY_INDEX = IndexConfig(chunker='session', embedder='ascii-hash',
                              hierarchy='metadata', summarizer='centroid')
# `metadata` rather than `louvain`: it groups by the storylines the corpus
# already declares, so it needs no graph, no vectors and no optional wheel, and
# it produces the same groups on every machine. What is under test is whether a
# summary row can be *seen*, which is independent of how the groups were found.


# Real in-memory index with a real hierarchy.
def test_every_row_of_a_hierarchical_index_is_visible_in_one_of_the_two_views():
    # this is an integration test
    """No row the build wrote may be absent from both views. A summary
    spanning more than one session gets `session_id=''`, and
    `chunks_by_session` iterates only chunks with a truthy session id — so
    without a strict partition, a multi-session summary is invisible while a
    single-session one leaks into the chunk view, indistinguishable from a
    diary entry."""
    index = IndexRegistry(LAB_SETTINGS, corpus.load_diary()).get(HIERARCHY_INDEX)
    leaves = [c for c in index.chunks if c.layer != 'summary']
    summaries = present.summary_rows(index)
    groups = present.chunks_by_session(index)

    assert summaries, 'this grouping produced no summary — nothing under test'
    assert len(index.chunks) > len(leaves), 'the build added no rows to see'

    # the partition: the two views together account for every row, once each
    assert sum(len(g['chunks']) for g in groups) == len(leaves)
    assert len(summaries) == len(index.chunks) - len(leaves)
    shown = {c['id'] for g in groups for c in g['chunks']}
    assert not (shown & {s['id'] for s in summaries}), \
        'a summary leaked into the chunk view, where it reads as a diary entry'

    # a summary says what it is, because its text alone does not: which group it
    # speaks for, at which level, and over how many chunks
    first = summaries[0]
    for key in ('id', 'text', 'group_id', 'level', 'members', 'member_ids',
                'sessions', 'chars'):
        assert key in first, f'missing {key}'
    assert first['text'] and first['group_id']
    assert first['level'] >= 1, 'a leaf is level 0; a summary is written above it'
    assert first['members'] == len(first['member_ids']) >= 1
    assert all(mid in index.by_id for mid in first['member_ids']), \
        'a summary must name members this index actually holds'

    # the case that was wholly invisible: a group spanning several sessions, which
    # is the normal shape of one and carries no session id at all
    spanning = [s for s in summaries if s['sessions'] > 1]
    assert spanning, 'expected at least one group to span several sessions'

    # a flat index has no summaries and says so with an empty list, not by
    # omitting the key — "no hierarchy" and "a hierarchy that found nothing" are
    # different facts, and only one of them is worth investigating
    flat = IndexRegistry(LAB_SETTINGS, corpus.load_diary()).get(
        IndexConfig(chunker='session', embedder='ascii-hash'))
    assert present.summary_rows(flat) == []


# A direct assert over the Inspector's job table and its own source, rather
# than a full chunks-job build and a poll for the absence of a database file
# — the round trip through the job runner is what
# `test_chunks_job_returns_sessions_and_any_summaries_beside_them` already
# proves.
def test_the_inspector_constructs_its_job_table_with_no_recorder():
    # this is a convention test
    """`Jobs(record=None)` — the default — is what keeps the Inspector's
    scratch builds for looking at chunks from becoming a second writer of the
    lab's experiment ledger: `record(job, state)` is called once per finished
    job, or nothing is, and the Inspector must be the "nothing" case. The
    lab's own `panel_server.py` passes `record=ledger.record` at its own
    construction site (checked against the same source below), so the
    Inspector's absence of an argument is the whole of the guard."""
    from raglab.dashboard.panel_server import Jobs
    assert Jobs().record is None, 'Jobs must default to no recorder'

    inspector_source = Path(inspector.__file__).read_text(encoding='utf-8')
    assert 'jobs = Jobs()' in inspector_source, (
        'the Inspector must construct its job table with no recorder')
    assert 'record=ledger.record' not in inspector_source, (
        "the Inspector must not adopt the lab's own recording call")

    from raglab.dashboard import panel_server as server
    server_source = Path(server.__file__).read_text(encoding='utf-8')
    assert 'Jobs(record=ledger.record)' in server_source, (
        'the lab, unlike the Inspector, does record — the contrast this '
        'guard depends on')


# FastAPI TestClient over the read-only app.
def test_config_endpoint_serves_the_chosen_config_and_the_labs_own_lists(monkeypatch):
    # this is an integration test
    """The frontend reads its fallback config from here rather than keeping
    its own copy. The option lists ride along, reused from `lab_config.py`
    rather than retyped — a list written twice is a list whose two readers
    eventually offer different pipelines."""
    from raglab.configuration import lab_config

    client = _client(monkeypatch)
    body = client.get('/api/config').json()

    assert body['chosen'] == inspector.CHOSEN_CONFIG
    assert body['chunkers'] == list(lab_config.CHUNKERS)
    assert body['embedders'] == list(lab_config.EMBEDDERS)
    assert body['retrievers'] == list(lab_config.RETRIEVERS)
    assert body['rerankers'] == list(lab_config.RERANKERS)
    assert body['graders'] == list(lab_config.GRADERS)
    # A config naming a value its own lists do not offer is the drift this route
    # exists to make impossible, so every field of the chosen config is checked
    # against the list it is chosen from.
    assert body['chosen']['index']['chunker'] in body['chunkers']
    assert body['chosen']['index']['embedder'] in body['embedders']
    assert body['chosen']['retrieval']['retriever'] in body['retrievers']
    assert body['chosen']['retrieval']['reranker'] in body['rerankers']
    assert body['chosen']['retrieval']['grader'] in body['graders']
