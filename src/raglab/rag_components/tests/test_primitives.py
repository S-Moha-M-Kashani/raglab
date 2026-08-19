"""Text normalisation, embedders, chunking, habits, query understanding,
retrieval primitives and metrics — the lab's building blocks, tested in
isolation from any index or pipeline."""
import numpy as np
import pytest

from raglab.rag_components.indexing import chunking_strategies as chunking
from raglab.llm_backends import cli_subprocess_chat as clichat
from raglab.corpora import diary_corpus_loader as corpus
from raglab.rag_components.indexing import embedding_backends as embedding
from raglab.evaluation import deterministic_metrics as metrics
from raglab.rag_components import question_to_answer_pipeline as pipeline
from raglab.rag_components.retrieval import query_understanding as query
from raglab.rag_components.retrieval import (
    retrieve_fuse_rerank_grade as retrieval)
from raglab.rag_components.retrieval import farsi_text_normalizer as textnorm
from raglab.configuration.lab_config import IndexConfig
from raglab.configuration.option_vocabularies import CHUNKERS


# --- text normalisation ----------------------------------------------------

@pytest.mark.parametrize('raw,expected', [
    # Persian/Arabic-Indic digits fold to ASCII.
    ('سال ۱۴۰۵', 'سال 1405'),
    # ي (Arabic yeh) and ك (Arabic kaf) fold to their Persian equivalents.
    ('يك', 'یک'),
    # Harakat (here a kasra) are stripped alongside the letterform fold.
    ('كِتاب', 'کتاب'),
    # Runs of spaces collapse to one, together with the digit fold.
    ('مي‌خواستم   بلاخره ۳ بار', 'می‌خواستم بلاخره 3 بار'),
])
def test_normalize_folds_equivalent_spellings_to_one_canonical_form(raw, expected):
    # this is a unit test
    """Two spellings a reader would call identical — Arabic letterforms,
    Persian vs. Arabic-Indic digits, decorative harakat, doubled spaces — must
    normalise to the exact same string, and normalising an already-canonical
    string a second time must not change it (`normalize`'s own idempotence
    guarantee)."""
    assert textnorm.normalize(raw) == expected
    assert textnorm.normalize(expected) == expected


@pytest.mark.parametrize('text,drop_stopwords,expected', [
    # A half-spaced compound is emitted whole *and* split, so it matches the
    # fully-spaced spelling the corpus also uses.
    ('می‌خوام برم باشگاه', True, {'میخوام', 'خوام', 'برم', 'باشگاه'}),
    ('می خوام برم باشگاه', True, {'خوام', 'برم', 'باشگاه'}),
    # Stopwords dropped by default...
    ('که از به پریا دعوا', True, {'پریا', 'دعوا'}),
    # ...but kept on request, for phrases like «ماه پیش» the stop list would
    # otherwise gut.
    ('که پریا', False, {'که', 'پریا'}),
    # Real content still comes through once the sentence's own stopwords
    # (با, شد) are dropped.
    ('امروز با پریا دعوام شد', True, {'امروز', 'پریا', 'دعوام'}),
    # Noise: nothing to tokenise, or nothing that survives the length/word
    # filters (a single Latin letter, a question mark, an underscore).
    ('', True, set()),
    ('a ؟ _', True, set()),
], ids=['half-space-joined-emits-both-forms', 'half-space-spaced-form',
        'stopwords-dropped-by-default', 'stopwords-kept-on-request',
        'real-content-survives-its-own-stopwords', 'empty-input',
        'pure-noise-filtered-to-nothing'])
def test_tokens_handle_half_space_compounds_stopwords_and_noise(text, drop_stopwords, expected):
    # this is a unit test
    """`tokens()` must emit a half-spaced compound both joined and split, obey
    `drop_stopwords`, and reduce pure noise to nothing — three behaviours a
    reader could otherwise mistake for three unrelated bugs."""
    assert set(textnorm.tokens(text, drop_stopwords)) == expected


@pytest.mark.parametrize('text,expected', [
    # «و بعدش» is a spoken-diary sentence boundary, same as the period.
    ('امروز رفتم سر کار و بعدش پریا زنگ زد. خیلی خسته بودم',
     ['امروز رفتم سر کار', 'پریا زنگ زد.', 'خیلی خسته بودم']),
    ('رفتم سر کار و بعدش پریا زنگ زد. خسته بودم',
     ['رفتم سر کار', 'پریا زنگ زد.', 'خسته بودم']),
    # «؟» is a boundary character alongside «.», «!».
    ('کجا رفتی؟ هیچ جا', ['کجا رفتی؟', 'هیچ جا']),
    # Whitespace-only text has no sentences at all.
    ('   ', []),
], ids=['run-on-marker-then-period-three-sentences',
        'run-on-marker-then-period-shorter-variant',
        'question-mark-boundary', 'whitespace-only-is-empty'])
def test_sentences_split_at_punctuation_and_spoken_run_on_markers(text, expected):
    # this is a unit test
    assert textnorm.sentences(text) == expected


def test_character_ngrams_share_a_stem_across_affixes():
    # this is a unit test
    assert set(textnorm.char_ngrams('میخواستم')) & set(textnorm.char_ngrams('نمیخواستم'))
    assert textnorm.char_ngrams('اب', 4) == ['اب']   # shorter than the window
    assert textnorm.char_ngrams('') == []


# --- embedders -------------------------------------------------------------

def test_ascii_hash_embedder_is_blind_to_farsi():
    # this is a unit test
    """An `[a-z0-9]+` tokeniser embeds a Farsi diary to the zero vector, so
    retrieval over it is arbitrary — the finding that moved the brain's
    default embedder off `hash`."""
    vectors = embedding.make_embedder('ascii-hash').embed(['امروز با پریا دعوام شد'])
    assert not np.any(vectors)


def test_char_hash_prefers_a_paraphrase_over_an_unrelated_line():
    # this is a unit test
    embedder = embedding.make_embedder('char-hash')
    vectors = embedder.embed(['دعوا با پریا سر کارهای خونه',
                              'باز با پریا دعوا کردیم سر خونه',
                              'نامه اداره مالیات رسید'])
    assert float(vectors[0] @ vectors[1]) > float(vectors[0] @ vectors[2])


def test_token_hash_is_normalised_and_nonzero_for_farsi():
    # this is a unit test
    vectors = embedding.make_embedder('token-hash').embed(['خواب بی‌خوابی کمردرد'])
    assert np.any(vectors)
    assert abs(float(np.linalg.norm(vectors[0])) - 1.0) < 1e-5


# --- chunking --------------------------------------------------------------

@pytest.mark.parametrize('chunker', CHUNKERS)
def test_every_chunker_yields_unique_nonempty_chunks_and_tracks_every_message(session, chunker):
    # this is a unit test
    """Every chunker, of every kind, must produce unique ids over nonempty
    text. `message`, `turn-pair` and `semantic-drift` compute `msg_start`/
    `msg_end` from where each chunk actually begins and ends, so no message
    may be dropped from that span — the ground truth cites evidence by
    message index — and the check below reads those fields back. `session`
    hard-codes the full span (`msg_start=0, msg_end=len(messages)-1`) on its
    one chunk regardless of what got emitted, so reading the same fields back
    would compare the index against itself; instead this checks that every
    message's own content actually landed in the emitted text. `fixed` and
    `fixed-overlap` chunk by character window and record no span at all
    (`msg_start`/`msg_end` stay -1), so neither claim applies to them."""
    cfg = IndexConfig(chunker=chunker, embedder='char-hash', contextual=False)
    embedder = embedding.make_embedder('char-hash')
    chunks = chunking.chunk_session(session, cfg, embedder)
    assert chunks, chunker
    assert len({c.id for c in chunks}) == len(chunks), chunker
    assert all(c.text.strip() for c in chunks), chunker
    if chunker == 'session':
        assert len(chunks) == 1, chunker
        for message in session['messages']:
            assert message['content'] in chunks[0].text, chunker
    elif chunker not in ('fixed', 'fixed-overlap'):
        covered = set()
        for chunk in chunks:
            covered.update(range(chunk.msg_start, chunk.msg_end + 1))
        assert covered == set(range(len(session['messages']))), chunker


def test_fixed_chunker_matches_the_production_packing(session):
    # this is a unit test
    """Same greedy 500-char packing the brain ships, or the comparison is
    against a straw man."""
    from raglab.rag_components.indexing.chunking_strategies import chunk_text
    cfg = IndexConfig(chunker='fixed', chunk_chars=500, contextual=False)
    ours = chunking.chunk_session(session, cfg, embedding.make_embedder('char-hash'))
    theirs = chunk_text(corpus.session_text(session), 500)
    assert [c.text for c in ours] == theirs


def test_contextual_prefix_situates_the_chunk(session):
    # this is a unit test
    cfg = IndexConfig(chunker='message', contextual=True)
    chunk = chunking.chunk_session(session, cfg, embedding.make_embedder('char-hash'))[0]
    assert session['date'] in chunk.prefix
    assert session['mood']['label'] in chunk.prefix
    assert chunk.body and not chunk.body.startswith('[')


def _long_session() -> dict:
    """A synthetic session with enough text to guarantee at least two
    fixed-overlap windows at chunk_chars=300/overlap=150 — the real corpus's
    `session` fixture is not guaranteed to be long enough, which used to make
    the overlap assertion below skip itself instead of running. Every word
    in the body is its own unique token (`واژه0001`, `واژه0002`, …) rather
    than a fixed phrase with a leading numeral varying — a shared numeral
    still leaves the *rest* of the phrase identical everywhere, which is
    exactly what let an earlier version of this fixture (one filler phrase
    repeated, then one phrase per sentence with only the number changing)
    satisfy a shared-substring check on every adjacent pair regardless of
    whether the windows actually overlapped. With no word repeated anywhere
    in the whole text, a substring shared between two chunks can only come
    from a window that genuinely spans the same stretch of source text."""
    words = [f'واژه{i:04d}' for i in range(1, 260)]
    midpoint = len(words) // 2
    return {'session_id': 'long-1', 'date': '2026-01-01', 'time': '21:00',
            'source': 'voice', 'mood': {'label': 'خسته', 'valence': 4, 'arousal': 5},
            'topics': [], 'recurring_threads': [],
            'messages': [{'role': 'user', 'intent': 'venting',
                          'content': ' '.join(words[:midpoint])},
                         {'role': 'assistant', 'content': ' '.join(words[midpoint:])}]}


def test_overlap_chunker_repeats_material_between_windows():
    # this is a unit test
    """Adjacent windows must share material, or `fixed-overlap` is `fixed`
    with extra config. Run over a synthetic session built long enough to
    window at least twice, rather than skipping when the corpus session
    handed to it happens to be short — a test that can skip itself is a test
    that can assert nothing."""
    session = _long_session()
    cfg = IndexConfig(chunker='fixed-overlap', chunk_chars=300, overlap=150,
                      contextual=False)
    chunks = chunking.chunk_session(session, cfg, embedding.make_embedder('char-hash'))
    assert len(chunks) >= 2, 'the synthetic session must be long enough to window'
    total = sum(len(c.text) for c in chunks)
    assert total > len(corpus.session_text(session))
    # Not just longer overall: the tail of each window has to reappear,
    # verbatim, at the head of the next one — every sentence here is
    # numbered uniquely, so this substring cannot be satisfied by chance.
    for a, b in zip(chunks, chunks[1:]):
        tail = ' '.join(a.text.split()[-3:])
        assert tail in b.text, (tail, b.text)


def test_semantic_drift_cuts_at_an_explicit_topic_shift():
    # this is a unit test
    fake = {'session_id': 'x-1', 'date': '2026-01-01', 'time': '22:00',
            'source': 'voice', 'mood': {'label': 'خسته', 'valence': 4, 'arousal': 5},
            'topics': [], 'recurring_threads': [],
            'messages': [
                {'role': 'user', 'intent': 'venting',
                 'content': 'امروز کل روز درگیر مالیات بودم و نامه اداره مالیات'},
                {'role': 'assistant', 'content': 'سخت بوده. چی شد آخرش؟'},
                {'role': 'user', 'intent': 'venting',
                 'content': 'حالا اینا رو ولش کن، پریا سر کارهای خونه دوباره دعوا کرد'},
                {'role': 'assistant', 'content': 'چه حسی داشتی؟'}]}
    cfg = IndexConfig(chunker='semantic-drift', chunk_chars=500, contextual=False)
    chunks = chunking.chunk_session(fake, cfg, embedding.make_embedder('char-hash'))
    assert len(chunks) >= 2
    assert any('پریا' in c.text and 'مالیات' not in c.text for c in chunks)


def test_chunk_metadata_is_chroma_safe(session):
    # this is a unit test
    cfg = IndexConfig(chunker='message')
    chunk = chunking.chunk_session(session, cfg, embedding.make_embedder('char-hash'))[0]
    for key, value in chunk.metadata().items():
        assert isinstance(value, (str, int, float, bool)), key


def test_importance_rises_with_emotional_intensity():
    # this is a unit test
    calm = {'mood': {'label': 'آروم', 'valence': 6, 'arousal': 2}}
    wrecked = {'mood': {'label': 'داغون', 'valence': 1, 'arousal': 9}}
    assert chunking.importance_of(wrecked) > chunking.importance_of(calm)


# --- habits: the card you repeat instead of finish -------------------------
# A board habit card carries `habitCount` repetitions per `habitFreq` period
# and a `habitHistory` of completions; the corpus declares them the same way.

HABIT_FREQS = ('daily', 'weekly', 'monthly', 'yearly')


def habit_period(freq: str, day: str) -> str:
    """The board's period id for a date, reimplemented from the spec rather
    than imported — a test that shares code with the thing it checks cannot
    catch the two drifting apart."""
    from datetime import date
    d = date.fromisoformat(day)
    if freq == 'yearly':
        return f'{d.year}'
    if freq == 'monthly':
        return f'{d.year}-{d.month:02d}'
    if freq == 'weekly':
        year, week, _ = d.isocalendar()
        return f'{year}-W{week:02d}'
    return d.isoformat()


def test_the_bundled_corpus_declares_consistent_habits(diary):
    # this is a convention test
    """Validates fixture data, not code: the habit corpus must be internally
    consistent the way a board habit card is (freq/count/history agree with
    each other), consistent with the sessions and thread the diarist's habit
    tracking is woven into, and varied enough (three of four cadences) that
    the period arithmetic for daily and monthly habits is exercised by every
    run, not just weekly ones. Folds five formerly separate data-validation
    tests, each of which validated fixture data rather than lab code."""
    habits = diary['habits']
    assert habits, 'the corpus must carry the habits the diarist tracks'
    for slug, habit in habits.items():
        assert habit['freq'] in HABIT_FREQS, slug
        assert habit['count'] >= 1, slug
        assert habit['title_fa'], slug
        assert isinstance(habit['times'], list), slug
        assert isinstance(habit['history'], dict), slug
        for period, days in habit['history'].items():
            # `habitHistory` is bucketed by period id, so a date filed under
            # the wrong bucket would make every count wrong.
            for day in days:
                assert habit_period(habit['freq'], day) == period, f'{slug} {day}'
            # The punch strip has exactly `count` boxes, so more completions
            # than that in one period could not have come from the board.
            assert len(days) <= habit['count'], f'{slug} {period}'
            assert len(days) == len(set(days)), f'{slug} {period} has a repeat'
    assert {h['freq'] for h in habits.values()} >= {'daily', 'weekly', 'monthly'}

    # Additive on purpose: appended on dates the corpus had not used, so
    # every pre-existing session stays exactly as it was.
    sessions = diary['sessions']
    ids = [s['session_id'] for s in sessions]
    assert len(ids) == len(set(ids)), 'a session id was reused'
    assert ids == sorted(ids), 'the corpus must stay chronological'
    habit_sessions = [s for s in sessions if 'habit-tracking' in s['recurring_threads']]
    assert len(habit_sessions) >= 8
    period = diary['meta']['period']
    for s in habit_sessions:
        assert period['from'] <= s['date'] <= period['to'], s['session_id']

    # thread_layer builds its digest title from this description; a thread
    # with no entry gets an empty one, which reads as a bug in the digest.
    assert diary['threads']['habit-tracking']


def test_every_chunk_reports_a_habit_field_even_when_it_has_none(session):
    # this is a unit test
    """Chroma metadata is a fixed shape per collection in practice: a field that
    only some rows carry turns a `where` clause into a silent partial scan."""
    assert 'habit-tracking' not in session['recurring_threads']
    chunk = chunking.chunk_session(session, IndexConfig(chunker='session'),
                                   None)[0]
    meta = chunk.metadata()
    assert meta['session_id'] == session['session_id']
    # `threads` is where a habit-tracking session shows up; the key is present
    # on every chunk even when, as here, the session tracks no habit.
    assert 'threads' in meta
    assert 'habit-tracking' not in meta['threads']


def test_habit_questions_cite_verbatim_evidence_like_the_rest_of_the_set(
        diary, ground_truth):
    # this is a unit test
    """The evidence quote is what quote-recall measures survival of, so a quote
    that is not literally in its cited message silently scores every config down."""
    sessions = corpus.sessions_by_id(diary)
    habit_questions = [q for q in ground_truth['questions']
                       if q['type'] == 'habit']
    assert len(habit_questions) >= 6
    for question in habit_questions:
        assert question['answer_fa'] and question['key_facts'], question['id']
        for ev in question['evidence']:
            messages = sessions[ev['session_id']]['messages']
            cited = ' '.join(messages[i]['content'] for i in ev['message_indices'])
            assert ev['quote'] in cited, f"{question['id']} {ev['session_id']}"


def test_every_question_type_is_one_the_report_breaks_down(ground_truth):
    # this is a unit test
    """metrics.aggregate walks TYPES, so a question type missing from it is
    dropped from the per-type table without any error — the breakdown just
    quietly stops covering part of the set."""
    assert {q['type'] for q in ground_truth['questions']} <= set(metrics.TYPES)


# --- query understanding ---------------------------------------------------

@pytest.mark.parametrize('question,expect_from,expect_to', [
    ('آذر چه خبر بود؟', 20251122, 20251221),
    ('پارسال پاییز حالم چطور بود؟', 20240923, 20241221),
    ('نوروز چی شد؟', 20260318, 20260404),
])
def test_time_scopes_resolve_to_the_right_window(question, expect_from, expect_to):
    # this is a unit test
    scope = query.resolve_time_scope(question, '2026-07-28')
    assert scope is not None, question
    assert (scope.from_int, scope.to_int) == (expect_from, expect_to)
    assert scope.label == {'آذر چه خبر بود؟': 'آذر', 'پارسال پاییز حالم چطور بود؟': 'پاییز پارسال',
                           'نوروز چی شد؟': 'نوروز'}[question]


def test_untimed_question_has_no_scope():
    # this is a unit test
    assert query.resolve_time_scope('چرا با پریا دعوا می‌کنیم؟', '2026-07-28') is None


def test_relative_month_scope_is_the_previous_calendar_month():
    # this is a unit test
    scope = query.resolve_time_scope('ماه پیش چی کار کردم؟', '2026-07-28')
    assert scope and (scope.from_int, scope.to_int) == (20260601, 20260630)


def test_where_clause_overlaps_rather_than_contains():
    # this is a unit test
    """A chunk whose span straddles the edge of the window is kept: a scope
    asks about a period, not that the evidence sit entirely inside it."""
    scope = query.TimeScope(20260101, 20260131, 'دی', 'jalali-month')
    clause = query.where_clause(scope)
    assert clause['$and'][0] == {'span_from': {'$lte': 20260131}}
    assert clause['$and'][1] == {'span_to': {'$gte': 20260101}}
    assert query.where_clause(None) is None


def test_expansion_adds_a_synonym_variant():
    # this is a unit test
    variants = query.expand('دعوا با همسرم سر چی بود؟')
    assert len(variants) >= 2
    assert any('پریا' in v for v in variants)


def test_keyword_query_strips_interrogatives():
    # this is a unit test
    assert 'چی' not in query.keyword_query('حال مامان چی شد؟')


# --- retrieval primitives --------------------------------------------------

def test_bm25_finds_the_document_with_the_rare_term():
    # this is a unit test
    bm25 = retrieval.BM25(['نامه اداره مالیات رسید و جریمه خوردم',
                           'با پریا دعوا کردیم', 'رفتم پیاده‌روی'])
    top = bm25.top('مالیات جریمه', 2)
    assert top and top[0][0] == 0


def test_bm25_respects_the_allowed_mask():
    # this is a unit test
    bm25 = retrieval.BM25(['مالیات', 'مالیات'])
    allowed = np.array([False, True])
    assert [i for i, _ in bm25.top('مالیات', 2, allowed)] == [1]


def test_rrf_ranks_a_document_both_retrievers_agree_on_first():
    # this is a unit test
    fused = retrieval.rrf([['a', 'b', 'c'], ['b', 'a', 'd']])
    assert max(fused, key=fused.get) in ('a', 'b')
    assert fused['a'] > fused['c'] and fused['b'] > fused['d']


def test_mmr_breaks_up_near_duplicates():
    # this is a unit test
    vectors = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    relevance = np.array([1.0, 0.99, 0.5], dtype=np.float32)
    assert retrieval.mmr(vectors, relevance, 2, 1.0) == [0, 1]
    assert retrieval.mmr(vectors, relevance, 2, 0.5) == [0, 2]


def test_mmr_falls_back_when_vectors_are_missing():
    # this is a unit test
    relevance = np.array([0.2, 0.9], dtype=np.float32)
    assert retrieval.mmr(np.zeros((0, 2), dtype=np.float32), relevance, 2, 0.5) == [1, 0]


def test_recency_weight_halves_after_one_half_life():
    # this is a unit test
    weight = retrieval.recency_weight(20260101, 20260701, 180.0)
    assert 0.4 < weight < 0.6


def test_llm_grade_parser_defaults_unscored_lines_to_neutral():
    # this is a unit test
    class Reply:
        content = '1: 8\nnonsense\n3: 0'

    class Provider:
        def invoke(self, messages, **kwargs):
            return Reply()

    scores = retrieval.llm_scores(Provider(), 'm', 'q', ['a', 'b', 'c'])
    assert scores[0] == pytest.approx(0.8)
    assert scores[1] == pytest.approx(0.5)   # unparsed = no opinion
    assert scores[2] == pytest.approx(0.0)


# --- metrics ---------------------------------------------------------------

def test_retrieval_metric_arithmetic():
    # this is a unit test
    retrieved, gold = ['a', 'x', 'b'], ['a', 'b', 'c']
    assert metrics.recall_at_k(retrieved, gold, 3) == pytest.approx(2 / 3)
    assert metrics.precision_at_k(retrieved, gold, 3) == pytest.approx(2 / 3)
    assert metrics.mrr(retrieved, gold) == 1.0
    assert metrics.hit_at_k(['x'], gold, 1) == 0.0
    assert metrics.ndcg_at_k(['a', 'b'], gold, 2) > metrics.ndcg_at_k(['x', 'a'], gold, 2)


@pytest.mark.parametrize('answer,expected', [
    # Something from the right session, but not the answering sentence: recall
    # needs the sentence itself, not merely that the session was retrieved.
    ('حرف‌های دیگری از همان نشست', 0.0),
    # The answering sentence, with irregular spacing a chunker's own
    # whitespace normalisation could introduce — both the sentence match and
    # its tolerance for whitespace noise are exercised by this one case.
    ('گفتم آذر  تموم   شد و از هیچ    شرکتی هیچ خبری نیس بعدش', 1.0),
])
def test_quote_recall_needs_the_exact_sentence_and_tolerates_whitespace(answer, expected):
    # this is a unit test
    question = {'evidence': [{'session_id': 's1', 'message_indices': [0],
                              'quote': 'آذر تموم شد و از هیچ شرکتی هیچ خبری نیس'}]}
    assert metrics.quote_recall(answer, question) == expected


def test_latest_state_session_is_the_newest_evidence():
    # this is a unit test
    question = {'evidence': [{'session_id': '2025-12-01-a'},
                             {'session_id': '2026-05-12-a'}]}
    assert metrics.latest_state_session(question) == '2026-05-12-a'


def test_aggregate_reports_per_type_and_a_headline():
    # this is a unit test
    rows = [
        {'id': 'q1', 'type': 'single-hop', 'difficulty': 'easy', 'answerable': True,
         'recall': 1.0, 'quote_recall': 1.0, 'ndcg': 1.0, 'hit': 1.0,
         'layers': ['chunk'], 'latency_ms': 5},
        {'id': 'q2', 'type': 'abstention', 'difficulty': 'hard', 'answerable': False,
         'abstained_correctly': 1.0, 'layers': [], 'latency_ms': 5},
    ]
    summary = metrics.aggregate(rows)
    assert summary['n_questions'] == 2
    assert summary['by_type']['single-hop']['recall'] == 1.0
    assert 0 < summary['overall']['headline'] <= 1.0


def test_an_answerer_that_could_not_be_reached_says_so_on_the_row():
    # this is a unit test
    """`pipeline._llm_answer` catches everything the model raises, returns the
    canonical refusal, and records the caught error on the outcome's own
    diagnostics — checked one layer down at `_llm_answer` itself, rather than
    through `metrics.score_question`, so a CliError or a timeout does not look
    exactly like "the diary is silent about that" unless something says
    otherwise."""
    class Unreachable:
        def invoke(self, messages, **kwargs):
            raise clichat.CliError('claude did not answer within 600s')

    outcome = pipeline.Outcome(question='امروز چه خبر بود؟', contexts=[])
    answer = pipeline._llm_answer(outcome, Unreachable(), 'sonnet')
    assert answer == pipeline.REFUSAL
    assert 'did not answer' in outcome.diagnostics['answer_error']

    # And a model that does answer leaves no such diagnostic, so its presence
    # means one thing.
    class Answered:
        content = 'یک جواب'

    class Working:
        def invoke(self, messages, **kwargs):
            return Answered()

    worked = pipeline.Outcome(question='امروز چه خبر بود؟', contexts=[])
    answer2 = pipeline._llm_answer(worked, Working(), 'sonnet')
    assert answer2 == 'یک جواب'
    assert 'answer_error' not in worked.diagnostics
