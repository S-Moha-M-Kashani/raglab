"""A second corpus, and the rules that keep a score meaning something.

Every retrieval finding this lab has produced is a finding *about* the Farsi
diary fixture. Some are obviously general (an embedder that cannot represent the
script scores at chance) and some obviously are not (the Farsi time-scope
filter), and with one corpus there was no way to tell which was which.

So the tests here are mostly about not confusing the two: an index built over one
corpus can never answer a question from another, a leaderboard never ranks across
corpora, and a dataset whose evidence does not hold is refused rather than
measured.
"""
import json

import pytest
from fastapi.testclient import TestClient

from raglab import datasets
from raglab.llm_tools import leaderboard
from raglab.config import IndexConfig

from conftest import _finished

BUNDLED = ('support-en', 'meetings-de', 'research-multihop', 'smoke-mini')


def _valid(**overrides) -> dict:
    """The smallest dataset that passes: two sessions, one question, one quote
    that is really in the message it cites."""
    quote = 'the roof was fixed on 3 March'
    payload = {
        'dataset': {'id': 'tiny-test', 'name': 'Tiny', 'language': 'en'},
        'sessions': [
            {'session_id': 's-1', 'date': '2026-03-04',
             'messages': [{'role': 'user',
                           'content': f'Good news — {quote}, at last.'},
                          {'role': 'assistant', 'content': 'Noted.'}]},
            {'session_id': 's-2', 'date': '2026-03-05',
             'messages': [{'role': 'user', 'content': 'Nothing to report.'}]},
        ],
        'questions': [
            {'id': 'q-1', 'type': 'single-hop', 'difficulty': 'easy',
             'answerable': True, 'question': 'When was the roof fixed?',
             'answer': 'On 3 March 2026.',
             'evidence': [{'session_id': 's-1', 'message_indices': [0],
                           'quote': quote}]},
        ],
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def imports_here(tmp_path, monkeypatch):
    monkeypatch.setenv('RAGLAB_DATASETS', str(tmp_path / 'datasets'))
    datasets.forget()
    yield tmp_path / 'datasets'
    datasets.forget()


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv('RAGLAB_DB', str(tmp_path / 'raglab.db'))
    monkeypatch.setenv('RAGLAB_DATASETS', str(tmp_path / 'datasets'))
    datasets.forget()
    from raglab.server import create_app
    return TestClient(create_app())


# --- the contract ----------------------------------------------------------

def test_a_dataset_that_meets_the_contract_has_nothing_to_report():
    # this is a unit test
    assert datasets.validate(_valid()) == []


def _quote_rewritten(payload: dict) -> dict:
    payload['questions'][0]['evidence'][0]['quote'] = 'the roof was fixed in April'
    return payload


def _session_missing(payload: dict) -> dict:
    payload['questions'][0]['evidence'][0]['session_id'] = 's-99'
    return payload


def _message_index_out_of_range(payload: dict) -> dict:
    payload['questions'][0]['evidence'][0]['message_indices'] = [7]
    return payload


@pytest.mark.parametrize('mutate, expected_substring', [
    (_quote_rewritten, 'verbatim'),
    (_session_missing, 'does not contain'),
    (_message_index_out_of_range, 'outside session'),
], ids=['quote-not-verbatim', 'session-missing', 'message-index-out-of-range'])
def test_a_dataset_that_breaks_one_rule_is_refused_naming_it(mutate,
                                                              expected_substring):
    # this is a unit test
    """Every lexical measurement in this lab — quote recall, the Inspector's
    green spans, the offline RAGAS context metrics — is computed against these
    evidence quotes, so a dataset that misquotes its own corpus would score
    *confidently* about text that was never there. Three ways the contract
    can be broken, refused rather than repaired, each naming what it refused."""
    payload = mutate(_valid())
    problems = datasets.validate(payload)
    assert any(expected_substring in problem for problem in problems), problems


def test_every_problem_is_reported_at_once_and_names_what_it_is_about():
    # this is a unit test
    """One problem per attempt is a slow loop over a 200-question corpus, and a
    message that does not name the question is not a message, it is a search."""
    payload = _valid()
    payload['questions'][0]['type'] = 'made-up'
    payload['questions'][0]['difficulty'] = 'trivial'
    payload['sessions'][1]['date'] = '4 March'
    problems = datasets.validate(payload)
    assert len(problems) >= 3
    assert any(p.startswith('q-1:') for p in problems)
    assert any(p.startswith('s-2:') for p in problems)


def test_an_unanswerable_question_needs_no_evidence():
    # this is a unit test
    """Abstention questions are the ones the corpus deliberately cannot answer:
    demanding evidence for them would make the failure mode the relevance gate
    exists for unmeasurable."""
    payload = _valid()
    payload['questions'].append({
        'id': 'q-2', 'type': 'abstention', 'difficulty': 'medium',
        'answerable': False, 'question': 'Who paid for the roof?'})
    assert datasets.validate(payload) == []


# --- the bundled samples ---------------------------------------------------

def test_the_bundled_datasets_meet_their_contract_and_cover_the_failure_modes():
    # this is a unit test
    """These four are reference points — the corpora a finding is checked
    against to tell "true of retrieval" from "true of Farsi diaries". A
    reference point nobody validated is a second unknown, and between them
    they have to offer more than one language, questions that need two
    sessions, and questions the corpus cannot answer — otherwise they are
    four spellings of the same test."""
    loaded = {}
    for name in BUNDLED:
        path = datasets.BUNDLED_DIR / f'{name}.json'
        assert path.exists(), f'{name} is missing from fixtures/corpus_groundtruth_datasets/'
        payload = json.loads(path.read_text(encoding='utf-8'))
        problems = datasets.validate(payload)
        assert problems == [], (name, problems)
        loaded[name] = payload
    languages = {d['dataset']['language'] for d in loaded.values()}
    assert len(languages) >= 2, languages
    types = {q['type'] for d in loaded.values() for q in d['questions']}
    assert {'multi-hop', 'aggregation', 'abstention', 'adversarial'} <= types
    for name, payload in loaded.items():
        assert any(not q.get('answerable', True) for q in payload['questions']), (
            f'{name} cannot measure whether a pipeline knows when to refuse')


def test_the_catalogue_leads_with_the_built_in_corpus():
    # this is a unit test
    """Every finding this lab has produced is about it, and it is the default: a list
    ordered by whatever sorted first would put it in an arbitrary place."""
    found = datasets.catalogue()
    assert found[0].id == datasets.BUILTIN
    assert found[0].source == 'builtin'
    ids = {d.id for d in found}
    assert set(BUNDLED) <= ids


# --- loading ---------------------------------------------------------------

def test_a_loaded_dataset_arrives_in_the_shape_the_lab_speaks():
    # this is a unit test
    """The contract says `question`; six modules and every stored run say
    `question_fa`. The loader is the one place that translates."""
    diary, ground_truth = datasets.load('smoke-mini')
    assert diary['sessions'] and ground_truth['questions']
    question = ground_truth['questions'][0]
    assert question['question_fa']
    assert question['query_date'] == ground_truth['meta']['query_date']
    session = diary['sessions'][0]
    # The fields the chunkers read directly, filled here so a corpus that is not
    # a diary does not have to carry five fields it has no use for.
    for field in ('time', 'source', 'mood', 'topics', 'recurring_threads'):
        assert field in session, field
    assert session['mood']['valence'] == 5.5, 'a corpus without moods is neutral'
    assert session['language'] == 'en'


def test_an_unknown_dataset_says_what_there_is():
    # this is a unit test
    with pytest.raises(ValueError) as raised:
        datasets.load('not-a-corpus')
    assert 'smoke-mini' in str(raised.value)


# --- the index cannot mix corpora ------------------------------------------

def test_the_dataset_is_part_of_the_fingerprint_and_the_blank_case_is_pinned():
    # this is a unit test
    """The one bug this feature could plausibly introduce, and the most
    expensive to notice late: an index built over one corpus handed a question
    from another. `''` is fingerprinted exactly as it was before the field
    existed — every run in `.runs/` records a collection name, and a new field
    in IndexConfig would otherwise rename them all."""
    assert (IndexConfig(dataset='smoke-mini').fingerprint()
            != IndexConfig().fingerprint())
    assert (IndexConfig(dataset='smoke-mini').fingerprint()
            != IndexConfig(dataset='support-en').fingerprint())
    # The literal is the value the field-less IndexConfig produced, read off
    # the code as it stood before this commit rather than off the code it
    # is checking.
    assert IndexConfig().fingerprint() == '804444ae65db'
    assert IndexConfig(dataset='').collection() == 'raglab-804444ae65db'


# Two corpora, one registry.
def test_an_index_is_never_shared_between_two_corpora(monkeypatch):
    # this is an integration test
    """crosses into IndexRegistry/build — an index over one corpus must never
    answer for another."""
    from raglab.config import LabSettings
    from raglab.index import IndexRegistry

    settings = LabSettings(llm_provider='fake')
    registry = IndexRegistry(settings)
    small = registry.get(IndexConfig(dataset='smoke-mini', chunker='session',
                                     embedder='token-hash', contextual=False))
    other = registry.get(IndexConfig(dataset='support-en', chunker='session',
                                     embedder='token-hash', contextual=False))
    assert small.stats.collection != other.stats.collection
    assert small.stats.chunks == 5, 'the smoke set is five sessions'
    assert other.stats.chunks == 20
    texts = ' '.join(chunk.text for chunk in small.chunks)
    assert 'espresso' in texts and 'Meridian' not in texts


def test_a_corpus_that_is_not_farsi_is_chunked_in_its_own_language():
    # this is a unit test
    """The speaker tags and the contextual header are prepended to text that
    gets embedded, so writing them in Farsi over an English corpus adds a
    constant foreign phrase to every vector."""
    from raglab.chunking import chunk_session
    diary, _ = datasets.load('smoke-mini')
    chunks = chunk_session(diary['sessions'][0],
                           IndexConfig(chunker='message', contextual=True),
                           embedder=None)
    text = chunks[0].text
    assert 'User:' in text and 'کاربر' not in text
    assert 'mood:' in text and 'حال' not in text


# --- the service -----------------------------------------------------------

def test_the_options_serve_the_catalogue(client):
    # this is an integration test
    """The HTTP half of `test_the_catalogue_leads_with_the_built_in_corpus`.
    `defaults['index']['dataset']` has no other assertion anywhere in the
    suite — test_server.py pins the other index defaults but not this one —
    so it is kept here even though the rest of this test asserts only
    presence (catalogue reachability, not defaults/help content in general)."""
    body = client.get('/api/options').json()
    ids = [d['id'] for d in body['datasets']]
    assert ids[0] == datasets.BUILTIN
    assert set(BUNDLED) <= set(ids)
    assert body['defaults']['index']['dataset'] == ''


def test_a_run_scores_the_questions_of_the_dataset_it_names(client, tmp_path,
                                                            monkeypatch):
    # this is an integration test
    """The half of the rule the fingerprint cannot enforce: the index comes from
    one file and the questions must come from the same one. Distinct from
    `tests/test_e2e.py`'s smoke run, which checks only the ledger's `dataset`
    field and a partial (`limit=3`) selection — this checks the job result and
    the saved run file both carry `dataset`, and that all six of the corpus's
    own question ids (and none other) come back for a full run."""
    from raglab import evaluate
    monkeypatch.setattr(evaluate, 'RUNS_DIR', tmp_path)
    started = client.post('/api/evaluations', json={
        'index': {'dataset': 'smoke-mini', 'chunker': 'session',
                  'embedder': 'token-hash'},
        'retrieval': {'k': 3}, 'generation': {'answerer': 'extractive'},
        'ragas_mode': 'off', 'label': 'smoke'})
    assert started.status_code == 202, started.text
    job = _finished(client, started.json()['job_id'])
    assert job['state'] == 'done', job.get('error')
    result = job['result']
    assert result['dataset'] == 'smoke-mini'
    assert result['summary']['n_questions'] == 6
    assert {row['id'] for row in result['rows']} == {
        'mini-001', 'mini-002', 'mini-003', 'mini-004', 'mini-005', 'mini-006'}
    saved = json.loads((tmp_path / f'{result["run_id"]}.json').read_text(
        encoding='utf-8'))
    assert saved['dataset'] == 'smoke-mini'


def test_a_dataset_is_imported_through_the_panel_and_becomes_selectable(client,
                                                                        tmp_path):
    # this is an integration test
    added = client.post('/api/datasets', json=_valid())
    assert added.status_code == 200, added.text
    assert added.json()['id'] == 'tiny-test'
    assert added.json()['source'] == 'imported'
    assert (tmp_path / 'datasets' / 'tiny-test.json').exists()
    ids = [d['id'] for d in client.get('/api/datasets').json()['datasets']]
    assert 'tiny-test' in ids
    # and it can be measured against straight away
    body = client.get('/api/questions?dataset=tiny-test').json()
    assert [q['id'] for q in body['questions']] == ['q-1']


def test_an_import_that_breaks_the_contract_is_refused_with_every_reason(client):
    # this is an integration test
    payload = _valid()
    payload['questions'][0]['evidence'][0]['quote'] = 'never said this'
    payload['questions'][0]['type'] = 'made-up'
    refused = client.post('/api/datasets', json=payload)
    assert refused.status_code == 400
    detail = refused.json()['detail']
    assert 'verbatim' in detail and 'made-up' in detail


def test_the_built_in_corpus_cannot_be_overwritten_by_an_import(client):
    # this is an integration test
    payload = _valid()
    payload['dataset']['id'] = datasets.BUILTIN
    refused = client.post('/api/datasets', json=payload)
    assert refused.status_code == 400
    assert 'built-in' in refused.json()['detail']


# --- the leaderboard -------------------------------------------------------

def test_the_leaderboard_never_ranks_across_corpora():
    # this is a unit test
    """Two corpora are not two configurations of one measurement: the questions
    differ, so the means are not of the same thing."""
    rows = [
        {'run_id': 'a', 'label': 'diary', 'dataset': 'diary-fa',
         'ragas_decision': 0.7, 'ragas_decision_stderr': 0.01,
         'selection': {'question_ids': ['q-1', 'q-2']},
         'judge': {'model': 'm', 'provider': 'p'}, 'n_questions': 2},
        {'run_id': 'b', 'label': 'support', 'dataset': 'support-en',
         'ragas_decision': 0.9, 'ragas_decision_stderr': 0.01,
         'selection': {'question_ids': ['q-1', 'q-2']},
         'judge': {'model': 'm', 'provider': 'p'}, 'n_questions': 2},
    ]
    groups = leaderboard.group(rows)
    assert len(groups) == 2
    assert {g.dataset for g in groups} == {'diary-fa', 'support-en'}
    for found in groups:
        assert leaderboard.verdict(found) == 'unranked', (
            'one row per corpus cannot beat anything')
    assert 'support-en' in leaderboard.markdown(groups)


def test_a_run_from_before_datasets_existed_is_the_built_in_corpus():
    # this is a unit test
    """Not a guess: it is the only corpus there was. Treating the blank as
    unknown would quarantine every row already on the leaderboard."""
    rows = [{'run_id': 'old', 'label': 'a', 'ragas_decision': 0.7,
             'selection': {'question_ids': ['q-1']},
             'judge': {'model': 'm', 'provider': 'p'}, 'n_questions': 1},
            {'run_id': 'new', 'label': 'b', 'dataset': 'diary-fa',
             'ragas_decision': 0.6, 'selection': {'question_ids': ['q-1']},
             'judge': {'model': 'm', 'provider': 'p'}, 'n_questions': 1}]
    groups = leaderboard.group(rows)
    assert len(groups) == 1, 'the old row and the new one are the same corpus'


# --- what the import control says the file must look like -------------------
#
# The importer refuses rather than repairs, so a corpus that does not meet the
# contract is a list of problems and a second attempt. The panel can spend that
# round trip or state the shape before a file is picked; there is a whole
# contract nobody on that screen knew to read, and nothing on screen said so.

def test_the_import_control_describes_a_shape_the_importer_really_enforces():
    # this is a unit test
    """Every top-level key and closed vocabulary the description names has to
    be one `validate` really refuses the absence of — a description beside a
    checker is a description that can drift from it."""
    from raglab import explain
    text = explain.topics()['run.dataset-file']
    for named in ('dataset', 'sessions', 'questions',      # the three keys
                  'id', 'name', 'language',                 # what a corpus is
                  'session_id', 'date', 'messages', 'role', 'content',
                  'type', 'difficulty', 'answerable', 'answer',
                  'evidence', 'message_indices', 'quote'):
        assert named in text, named
    # The two closed vocabularies are named rather than gestured at, because the
    # importer refuses a value outside them and nobody can guess eleven types.
    # Listed by hand and pinned here: config cannot import metrics.
    from raglab.config import DIFFICULTIES
    from raglab.metrics import TYPES
    for value in TYPES + DIFFICULTIES:
        assert value in text, value
    # The rule that earns its cost is the one a reader must not discover from a
    # rejection, and the full contract is one line away.
    assert 'verbatim' in text
    assert 'fixtures/corpus_groundtruth_datasets/' in text
    # And each of the three top-level keys the text calls out is one
    # `validate` genuinely refuses the absence of.
    for key in ('dataset', 'sessions', 'questions'):
        without = _valid()
        without.pop(key)
        problems = datasets.validate(without)
        assert any(f'"{key}"' in p for p in problems), (
            f'{key} is described as required, but no problem names it: {problems}')


def test_the_panel_renders_the_json_shape_as_a_shape():
    # this is a convention test
    """This is the one help text with a structure in it, and a structure that
    arrives as one run-on paragraph is not one. Every other explainer is a
    single line, so preserving the newlines costs them nothing."""
    from raglab.server import STATIC
    css = (STATIC / 'panel.css').read_text(encoding='utf-8')
    rule = css.split('p.explain {')[1].split('}')[0]
    assert 'pre-wrap' in rule


def test_the_panel_offers_the_dataset_and_ranks_per_corpus():
    # this is a convention test
    from raglab.server import STATIC
    html = (STATIC / 'index.html').read_text(encoding='utf-8')
    js = (STATIC / 'panel.js').read_text(encoding='utf-8')
    assert 'id="dataset"' in html and 'id="dataset-file"' in html
    assert '/api/datasets' in js
    assert 'One table per dataset' in html
    assert 'byDataset' in js, 'the leaderboard renders one table per corpus'
