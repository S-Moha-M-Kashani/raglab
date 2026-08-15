import pytest

from raglab import evaluate, pipeline, corpus, inspector, present
from raglab.config import IndexConfig, RetrievalConfig, LabSettings
from raglab.index import IndexRegistry

LAB_SETTINGS = LabSettings(openrouter_api_key='', llm_provider='fake')


def test_evidence_spans_locate_the_quote_and_never_invent_one():
    """The green highlight is drawn from these ranges, so a range that is not
    really the quote is a lie on screen. `mark_gold` also calls a chunk
    *contained by* a quote gold, which has no verbatim quote inside it and
    must highlight nothing."""
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


# Real fixture, real chunker.
def test_a_question_reports_how_many_gold_chunks_existed_to_find():
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


def test_mark_gold_matches_evidence_quote_either_direction():
    quotes = ['قسط‌بندی جریمه اوکی شد شیش قسط']
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


# Real in-memory index, offline.
def test_chunks_by_session_groups_and_counts():
    diary = corpus.load_diary()
    index = IndexRegistry(LAB_SETTINGS, diary).get(
        IndexConfig(chunker='session', embedder='ascii-hash'))
    groups = inspector.chunks_by_session(index)

    assert len(groups) == len(index.by_session)
    total = sum(len(g['chunks']) for g in groups)
    assert total == len(index.chunks)
    first = groups[0]
    assert first['session_id'] and 'date' in first
    assert all('id' in c and 'text' in c for c in first['chunks'])


import time

from fastapi.testclient import TestClient


def _client(monkeypatch):
    from raglab import inspector
    # Pin the offline/fake backend so no test needs a key or a network.
    monkeypatch.setattr(inspector, 'load_lab_settings', lambda: LAB_SETTINGS)
    return TestClient(inspector.create_inspector_app())


# FastAPI TestClient over the read-only app.
def test_groundtruth_endpoint_returns_full_pairs(monkeypatch):
    client = _client(monkeypatch)
    body = client.get('/api/groundtruth').json()
    q = body['questions'][0]
    # the fields the :9002 /api/questions endpoint strips must be present here
    for key in ('answer_fa', 'key_facts', 'evidence', 'question_fa',
                'type', 'difficulty', 'answerable'):
        assert key in q, f'missing {key}'
    assert 'quote' in q['evidence'][0]


# FastAPI TestClient over the read-only app; real in-memory index build via
# the job runner.
def test_chunks_job_returns_sessions(monkeypatch):
    client = _client(monkeypatch)
    cfg = {'index': {'chunker': 'session', 'embedder': 'ascii-hash'}}
    acc = client.post('/api/chunks', json=cfg)
    assert acc.status_code == 202
    job_id = acc.json()['job_id']
    # jobs run on a daemon thread; poll until done
    for _ in range(200):
        job = client.get(f'/api/jobs/{job_id}').json()
        if job['state'] in ('done', 'error'):
            break
        time.sleep(0.02)
    assert job['state'] == 'done', job.get('error')
    result = job['result']
    assert result['total'] == sum(len(g['chunks'])
                                  for g in result['chunks_by_session'])


# FastAPI TestClient over the read-only app; real in-memory index build and
# retrieval trace via the job runner.
def test_trace_job_marks_gold(monkeypatch):
    client = _client(monkeypatch)
    gt_q = client.get('/api/groundtruth').json()['questions'][0]
    payload = {'index': {'chunker': 'session', 'embedder': 'ascii-hash'},
               'retrieval': {'retriever': 'hybrid-rrf', 'reranker': 'none',
                             'grader': 'none', 'k': 3, 'rerank_depth': 20,
                             'time_filter': False},
               'question_id': gt_q['id']}
    acc = client.post('/api/trace', json=payload)
    assert acc.status_code == 202
    job_id = acc.json()['job_id']
    for _ in range(200):
        job = client.get(f'/api/jobs/{job_id}').json()
        if job['state'] in ('done', 'error'):
            break
        time.sleep(0.02)
    assert job['state'] == 'done', job.get('error')
    cands = job['result']['trace']['candidates']
    assert cands and all('gold' in c for c in cands)


# An invalid config must refuse synchronously, the same as /api/questions,
# rather than accept the job and fail it with state='error'.
def test_trace_rejects_an_unknown_reranker(monkeypatch):
    client = _client(monkeypatch)
    gt_q = client.get('/api/groundtruth').json()['questions'][0]
    payload = {'index': {'chunker': 'session', 'embedder': 'ascii-hash'},
               'retrieval': {'reranker': 'nope'},
               'question_id': gt_q['id']}
    res = client.post('/api/trace', json=payload)
    assert res.status_code == 400
    assert 'unknown reranker' in res.json()['detail']


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
    }


# (file, must_contain, must_not_contain, reason) — one row per retired
# single-substring pin test, each carrying the one line that used to be its
# docstring so a failure names the rule rather than printing a bare
# "assert 'x' in text".
INSPECTOR_CONVENTIONS = [
    ('inspector.html', 'tab-groundtruth', None,
     'the served shell must expose its ground-truth tab hook'),
    ('inspector.html', 'tab-chunks', None,
     'the served shell must expose its chunks tab hook'),
    ('inspector.html', 'tab-retrieval', None,
     'the served shell must expose its retrieval tab hook'),
    ('inspector.html', 'inspector-tab', None,
     'the served shell must expose the tab-switching hook'),
    ('inspector.html', 'retrieval-table', None,
     'the served shell must expose the retrieval table hook'),
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
    ('inspector.html', 'question-picker', None,
     'the question picker must expose its own hook'),
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
    ('inspector.html', 'chunks-mode', None,
     'the chunks/summaries toggle must expose its own hook'),
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
    ('inspector.js', '/api/config', None,
     'inspector.js must fetch the config it renders, rather than assume one'),
    ('inspector.js', 'FOLLOWED_CONFIG', None,
     'following the lab must stay the primary path; the served config is only '
     'the fallback for a lab that is down'),
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


def test_the_inspector_shares_one_token_sheet_and_one_script_with_the_panel():
    # this is a convention test
    """`tokens.css` and `lab.js` are one file for both pages rather than a
    copy each, so a design token or a utility cannot drift apart on either
    page. This pins that the Inspector actually routes them, its page
    actually loads them, and each loads before the page's own stylesheet or
    script — a later link would lose the tokens to the page's own overrides
    instead of feeding them. The panel's half of this claim lives in
    test_panel.py."""
    from fastapi.testclient import TestClient

    client = TestClient(inspector.create_inspector_app())
    assert (inspector.STATIC / 'tokens.css').exists()
    assert (inspector.STATIC / 'lab.js').exists()

    html = client.get('/').text
    tokens = client.get('/tokens.css')
    lab = client.get('/lab.js')
    assert tokens.status_code == 200
    assert tokens.headers['content-type'].startswith('text/css')
    assert lab.status_code == 200
    assert lab.headers['content-type'].startswith('application/javascript')
    assert (html.index('href="/tokens.css"') < html.index('href="/inspector.css"'))
    assert (html.index('src="/lab.js"') < html.index('src="/inspector.js"'))


# --- following the lab (:9002) ----------------------------------------------

import json as _json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

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

# Newest first, the order the lab's own /api/jobs uses. Tests reassign this to
# say which run happened last.
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
    so the follow test is fast, offline and independent of the real lab's own
    behaviour."""
    server = ThreadingHTTPServer(('127.0.0.1', 0), _FakeLabHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}'
    finally:
        server.shutdown()
        thread.join(timeout=2)


# FastAPI TestClient; the lab it points at is an unreachable port, pinning "a
# lab that is not running is a normal state".
def test_follow_reports_lab_down_without_raising(monkeypatch):
    monkeypatch.setenv('RAGLAB_INSPECTOR_LAB_URL', 'http://127.0.0.1:9')
    client = _client(monkeypatch)
    res = client.get('/api/follow')
    assert res.status_code == 200
    body = res.json()
    assert body['lab'] == 'down'
    assert body['index'] is None and body['query'] is None


# FastAPI TestClient over the read-only app; the lab is a canned fake HTTP
# server, not the real :9002.
def test_follow_reads_a_finished_index_and_query_job(monkeypatch, fake_lab):
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


# FastAPI TestClient; the lab is a canned fake.
def test_follow_shows_one_table_per_selected_question(monkeypatch, fake_lab,
                                                      request):
    """The retrieval window must show *only* the questions the experiment
    picked. Both the retrieval-only run and a judged evaluation feed it, so
    `/api/follow` normalises them to one shape and the page keeps one
    renderer."""
    module = request.module
    monkeypatch.setenv('RAGLAB_INSPECTOR_LAB_URL', fake_lab)
    client = _client(monkeypatch)

    monkeypatch.setattr(module, 'FAKE_ORDER',
                        [FAKE_RETRIEVE_JOB, FAKE_RUN_JOB, FAKE_INDEX_JOB])
    view = client.get('/api/follow').json()['retrieval']
    assert view['kind'] == 'retrieve'
    assert view['config']['retrieval']['k'] == 3
    assert [q['question_id'] for q in view['questions']] == ['q-001', 'q-002']
    candidate = view['questions'][0]['trace']['candidates'][0]
    assert candidate['gold'] is True and candidate['fused_rank'] == 1

    # an evaluation that finished later is what the window follows instead, and
    # its traces arrive under a different key on the lab's side
    monkeypatch.setattr(module, 'FAKE_ORDER',
                        [FAKE_RUN_JOB, FAKE_RETRIEVE_JOB, FAKE_INDEX_JOB])
    view = client.get('/api/follow').json()['retrieval']
    assert view['kind'] == 'run'
    assert [q['question_id'] for q in view['questions']] == ['q-009']

    # no set-wide run at all is a normal state, not an error
    monkeypatch.setattr(module, 'FAKE_ORDER', [FAKE_INDEX_JOB])
    body = client.get('/api/follow').json()
    assert body['lab'] == 'up' and body['retrieval'] is None


# FastAPI TestClient; the lab is a canned fake.
def test_follow_shows_the_chunks_the_last_run_actually_used(monkeypatch,
                                                           fake_lab, request):
    """The two windows must describe the same pipeline. An evaluation builds
    its index *implicitly*, so it creates no index job — the chunks window
    must not keep showing whatever `Build` was last pressed while the
    retrieval window shows the experiment. So the chunks come from the
    newest job that produced any, whatever its kind."""
    module = request.module
    monkeypatch.setenv('RAGLAB_INSPECTOR_LAB_URL', fake_lab)
    client = _client(monkeypatch)

    # the exact shape that misled: the run is newer than the index build, and
    # they name different chunkers
    monkeypatch.setattr(module, 'FAKE_ORDER', [FAKE_RUN_JOB, FAKE_INDEX_JOB])
    body = client.get('/api/follow').json()
    assert body['index']['config']['index']['chunker'] == 'semantic-drift'
    assert body['index']['chunks_by_session'][0]['session_id'] == 'run-s1'
    # and both windows now agree about which index produced what is on screen
    assert (body['index']['config']['index']['chunker']
            == body['retrieval']['config']['index']['chunker'])

    # an explicit build afterwards is the newest again, and wins
    monkeypatch.setattr(module, 'FAKE_ORDER', [FAKE_INDEX_JOB, FAKE_RUN_JOB])
    body = client.get('/api/follow').json()
    assert body['index']['config']['index']['chunker'] == 'session'


# FastAPI TestClient over the read-only app; a real in-memory index, the
# offline embedder and the extractive answerer.
def test_adding_a_question_produces_rows_identical_to_the_run_s_own(monkeypatch):
    """A question you add by hand has to arrive scored exactly like the ones
    the experiment selected — same retrieval row, same generation row, same
    metric keys — or the two cannot be read side by side. The row it returns
    is asserted against what `metrics.score_question` produces for the same
    outcome."""
    from raglab import metrics
    client = _client(monkeypatch)
    gt_q = client.get('/api/groundtruth').json()['questions'][0]
    config = {'index': {'chunker': 'fixed-overlap', 'chunk_chars': 500,
                        'overlap': 100, 'contextual': True,
                        'embedder': 'ascii-hash'},
              'retrieval': {'retriever': 'hybrid-rrf', 'reranker': 'lexical',
                            'grader': 'none', 'k': 5, 'rerank_depth': 20,
                            'time_filter': False, 'multi_query': False},
              'generation': {'answerer': 'extractive'}}

    acc = client.post('/api/questions', json={**config, 'question_id': gt_q['id']})
    assert acc.status_code == 202, acc.text
    job = _wait(client, acc.json()['job_id'])
    assert job['state'] == 'done', job.get('error')
    result = job['result']

    # the retrieval half, shaped like a followed question
    retrieval = result['retrieval']
    assert retrieval['question_id'] == gt_q['id']
    assert isinstance(retrieval['gold_available'], int)
    candidate = retrieval['trace']['candidates'][0]
    for key in ('dense_rank', 'bm25_rank', 'fused_rank', 'rerank_score',
                'grade_score', 'kept', 'gold', 'gold_spans'):
        assert key in candidate, f'missing {key}'

    # the generation half, shaped like an evaluation's row: same keys, so the
    # added question shows the same metrics and no others
    row = result['generation']
    reference = metrics.score_question(
        gt_q, _outcome_for(config, gt_q), config['retrieval']['k'])
    assert set(row) == set(reference), (
        f"added row differs: only here {set(row) - set(reference)}, "
        f"only in the eval row {set(reference) - set(row)}")
    assert row['id'] == gt_q['id'] and row['answer']

    # and it says which config produced it, because a row measured under other
    # settings than its neighbours is worse than no row
    assert result['config']['index']['chunker'] == 'fixed-overlap'

    # an unknown id refuses synchronously rather than dying inside a job
    assert client.post('/api/questions',
                       json={**config, 'question_id': 'q-nope'}).status_code == 404


def _wait(client, job_id: str, tries: int = 400) -> dict:
    for _ in range(tries):
        job = client.get(f'/api/jobs/{job_id}').json()
        if job['state'] in ('done', 'error'):
            return job
        time.sleep(0.02)
    raise AssertionError(f'job {job_id} never finished')


def _outcome_for(config: dict, question: dict):
    """Computed here so the metric keys are compared against a real outcome
    rather than a guess."""
    from raglab.config import GenerationConfig, LabConfig
    cfg = LabConfig.from_dict(config)
    index = IndexRegistry(LAB_SETTINGS, corpus.load_diary()).get(cfg.index)
    gt = corpus.load_ground_truth()
    outcome = pipeline.retrieve(index, cfg.retrieval, question['question_fa'],
                                gt['meta']['query_date'])
    return pipeline.answer(outcome, GenerationConfig(answerer='extractive'))



# FastAPI TestClient over the read-only app.
def test_explain_serves_the_same_metric_help_the_lab_does(monkeypatch):
    """Served from `explain` — the lab's own source for /api/options —
    rather than copied into the Inspector's page, so the two panels cannot
    end up explaining the same metric differently."""
    from raglab import explain
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


# FastAPI TestClient; the lab is a canned fake.
def test_follow_exposes_what_the_evaluation_generated(monkeypatch, fake_lab,
                                                     request):
    """What the model wrote and how it scored come from the evaluation's own
    rows; only an evaluation has them, so a retrieval-only run leaves this
    `None` rather than showing a stale answer beside fresh ranks."""
    module = request.module
    monkeypatch.setenv('RAGLAB_INSPECTOR_LAB_URL', fake_lab)
    client = _client(monkeypatch)

    monkeypatch.setattr(module, 'FAKE_ORDER', [FAKE_RUN_JOB, FAKE_INDEX_JOB])
    view = client.get('/api/follow').json()['generation']
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
    monkeypatch.setattr(module, 'FAKE_ORDER', [FAKE_RETRIEVE_JOB, FAKE_INDEX_JOB])
    body = client.get('/api/follow').json()
    assert body['generation'] is None
    assert body['retrieval']['kind'] == 'retrieve'   # retrieval still follows


# --- summaries: the rows a hierarchy adds beside the leaves -----------------

HIERARCHY_INDEX = IndexConfig(chunker='session', embedder='ascii-hash',
                              hierarchy='metadata', summarizer='centroid')
# `metadata` rather than `louvain`: it groups by the storylines the corpus
# already declares, so it needs no graph, no vectors and no optional wheel, and
# it produces the same groups on every machine. What is under test is whether a
# summary row can be *seen*, which is independent of how the groups were found.


# Real in-memory index with a real hierarchy.
def test_every_row_of_a_hierarchical_index_is_visible_in_one_of_the_two_views():
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


# FastAPI TestClient over the read-only app; a real in-memory hierarchical
# build via the job runner.
def test_chunks_job_returns_the_summaries_beside_the_chunk_groups(monkeypatch):
    """The manual build path serves both halves in one job, so the toggle
    needs no second request — and `total` keeps counting leaves, since that
    is what the chunk-size knob is read against."""
    client = _client(monkeypatch)
    acc = client.post('/api/chunks', json={
        'index': {'chunker': 'session', 'embedder': 'ascii-hash',
                  'hierarchy': 'metadata', 'summarizer': 'centroid'}})
    assert acc.status_code == 202
    job = _wait(client, acc.json()['job_id'])
    assert job['state'] == 'done', job.get('error')
    result = job['result']

    assert result['total'] == sum(len(g['chunks'])
                                  for g in result['chunks_by_session'])
    assert result['total_summaries'] == len(result['summaries']) >= 1
    summary = result['summaries'][0]
    assert summary['members'] >= 1 and summary['level'] >= 1 and summary['text']


# FastAPI TestClient; the lab is a canned fake.
def test_follow_carries_the_summaries_the_lab_built(monkeypatch, fake_lab,
                                                   request):
    """The followed view has no index of its own — it reads what the lab's
    job reported — so a summary is visible on :9003 only if the lab put it
    on the job. A job from before this existed carries no such key, and must
    arrive as an empty list rather than an error."""
    module = request.module
    monkeypatch.setenv('RAGLAB_INSPECTOR_LAB_URL', fake_lab)
    client = _client(monkeypatch)

    with_summaries = {
        'id': 'idx-fake-2', 'kind': 'index', 'state': 'done',
        'config': {'index': {'chunker': 'session', 'hierarchy': 'metadata'}},
        'result': {'chunks': 2, 'chunks_by_session': [
            {'session_id': 's1', 'date': '2026-01-01',
             'chunks': [{'id': 's1-0', 'text': 'chunk one'}]}],
            'summaries': [{'id': 'summary:h1-000', 'text': 'a group card',
                           'group_id': 'h1-000', 'level': 1, 'members': 2,
                           'member_ids': ['s1-0', 's2-0'], 'sessions': 2,
                           'chars': 12}]}}

    monkeypatch.setattr(module, 'FAKE_ORDER', [with_summaries])
    view = client.get('/api/follow').json()['index']
    assert view['chunks_by_session'][0]['session_id'] == 's1'
    assert len(view['summaries']) == 1
    assert view['summaries'][0]['group_id'] == 'h1-000'
    assert view['summaries'][0]['sessions'] == 2

    # a flat build, and every job recorded before summaries were reported at all
    monkeypatch.setattr(module, 'FAKE_ORDER', [FAKE_INDEX_JOB])
    view = client.get('/api/follow').json()['index']
    assert view['chunks_by_session'][0]['session_id'] == 's1'
    assert view['summaries'] == []


# FastAPI TestClient over the read-only app; a real chunk build and a real
# SQLite file on a temp path.
def test_the_inspector_writes_nothing_to_the_labs_ledger(monkeypatch, tmp_path):
    """It builds its own in-memory index through the *lab's* job runner
    (`from .server import Jobs`), so when every finished job started
    recording itself into `Jobs.run`, the Inspector risked silently becoming
    a second writer of the lab's experiment ledger from a second OS process.
    A scratch build for looking at chunks is not an experiment anybody
    ranks."""
    from raglab import ledger

    db = tmp_path / 'raglab.db'
    monkeypatch.setenv('RAGLAB_DB', str(db))
    client = _client(monkeypatch)
    acc = client.post('/api/chunks', json={
        'index': {'chunker': 'session', 'embedder': 'ascii-hash'}})
    assert acc.status_code == 202
    job_id = acc.json()['job_id']
    for _ in range(400):
        job = client.get(f'/api/jobs/{job_id}').json()
        if job['state'] in ('done', 'error'):
            break
        time.sleep(0.02)
    assert job['state'] == 'done', job.get('error')

    # The build worked; it simply left no trace in the lab's record.
    assert job['result']['total'] > 0
    # Existence first, and only then a read: `ledger.connect` creates the schema,
    # so asking the ledger what is in it would bring the file into being and the
    # stronger assertion would pass for the wrong reason.
    assert not db.exists(), 'the Inspector must not even create the file'
    assert ledger.experiments(path=db) == []
    # And it did not quietly fail to record either — that would be the same bug
    # wearing an error message.
    assert 'ledger_error' not in job


# A build on a corpus that is not the built-in diary — the shape of the fault
# this pins: the lab names its dataset on every job it starts, and the Inspector
# reads the fixture from somewhere else entirely.
FAKE_OTHER_CORPUS_JOB = {
    'id': 'idx-fake-2', 'kind': 'index', 'state': 'done',
    'config': {'index': {'chunker': 'session', 'embedder': 'ascii-hash',
                         'dataset': 'meetings-de'}},
    'result': {'chunks': 1, 'chunks_by_session': [
        {'session_id': 'mtg-0113', 'date': '2026-01-13',
         'chunks': [{'id': 'mtg-0113:c0', 'text': 'Protokoll'}]}]}}


# FastAPI TestClient; the lab is a canned fake.
def test_follow_names_the_corpus_the_lab_is_working_on(monkeypatch, fake_lab,
                                                       request):
    """Which corpus the lab is on is a fact about the whole page, not about
    one window, so `/api/follow` answers it once rather than each window
    guessing separately."""
    module = request.module
    monkeypatch.setenv('RAGLAB_INSPECTOR_LAB_URL', fake_lab)
    client = _client(monkeypatch)

    monkeypatch.setattr(module, 'FAKE_ORDER',
                        [FAKE_OTHER_CORPUS_JOB, FAKE_INDEX_JOB])
    assert client.get('/api/follow').json()['dataset'] == 'meetings-de'

    # The newest job wins, exactly as every other window on this page follows
    # the newest job — switching back to the diary must switch the fixture back.
    monkeypatch.setattr(module, 'FAKE_ORDER',
                        [FAKE_INDEX_JOB, FAKE_OTHER_CORPUS_JOB])
    assert client.get('/api/follow').json()['dataset'] == ''

    # A job whose config names no index at all cannot say which corpus it ran
    # on, so it is passed over rather than read as the built-in one: "does not
    # say" and "says the diary" are different facts.
    monkeypatch.setattr(module, 'FAKE_ORDER',
                        [FAKE_QUERY_JOB, FAKE_OTHER_CORPUS_JOB])
    assert client.get('/api/follow').json()['dataset'] == 'meetings-de'

    # And a lab that is not running names no corpus rather than the diary.
    monkeypatch.setenv('RAGLAB_INSPECTOR_LAB_URL', 'http://127.0.0.1:9')
    assert _client(monkeypatch).get('/api/follow').json()['dataset'] == ''


# FastAPI TestClient over the read-only app.
def test_config_endpoint_serves_the_chosen_config_and_the_labs_own_lists(monkeypatch):
    """The frontend reads its fallback config from here rather than keeping
    its own copy. The option lists ride along, reused from `config.py`
    rather than retyped — a list written twice is a list whose two readers
    eventually offer different pipelines."""
    from raglab import config as lab_config

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
