"""Text normalisation, embedders, chunking, habits, query understanding,
retrieval primitives and metrics — the lab's building blocks, tested in
isolation from any index or pipeline."""
import numpy as np
import pytest

from raglab import chunking, clichat, corpus, embedding, metrics, pipeline, query, retrieval, textnorm
from raglab.config import IndexConfig


# --- text normalisation ----------------------------------------------------

def test_normalize_folds_arabic_letterforms_and_digits():
    assert textnorm.normalize('يك') == textnorm.normalize('یک')
    assert '۱۴۰۵' not in textnorm.normalize('سال ۱۴۰۵')
    assert '1405' in textnorm.normalize('سال ۱۴۰۵')


def test_normalize_is_idempotent():
    once = textnorm.normalize('مي‌خواستم   بلاخره ۳ بار')
    assert textnorm.normalize(once) == once


def test_tokens_match_across_half_space_spelling():
    joined = set(textnorm.tokens('می‌خوام برم باشگاه'))
    spaced = set(textnorm.tokens('می خوام برم باشگاه'))
    assert joined & spaced
    assert 'باشگاه' in joined and 'باشگاه' in spaced


def test_tokens_drop_stopwords_but_keep_content():
    tokens = textnorm.tokens('که از به پریا دعوا')
    assert 'پریا' in tokens and 'دعوا' in tokens
    assert 'که' not in tokens


def test_sentences_split_spoken_run_ons():
    text = 'امروز رفتم سر کار و بعدش پریا زنگ زد. خیلی خسته بودم'
    assert len(textnorm.sentences(text)) >= 2


def test_two_spellings_of_the_same_word_normalise_alike():
    # Arabic ي/ك, Persian digits and diacritics are rendering differences,
    # not word differences.
    assert textnorm.normalize('يك') == textnorm.normalize('یک')
    assert '1405' in textnorm.normalize('سال ۱۴۰۵')
    assert textnorm.normalize('كِتاب') == 'کتاب'
    once = textnorm.normalize('مي‌خواستم  ۳ بار')
    assert textnorm.normalize(once) == once      # called twice in places


def test_farsi_text_produces_tokens_and_noise_does_not():
    assert textnorm.tokens('امروز با پریا دعوام شد')
    assert textnorm.tokens('') == []
    assert textnorm.tokens('a ؟ _') == []       # single letters and punctuation


def test_stopwords_can_be_kept_for_the_time_filter():
    # It matches phrases like «ماه پیش», whose words the stop list removes.
    assert 'که' in textnorm.tokens('که پریا', drop_stopwords=False)


def test_a_half_spaced_compound_matches_the_spaced_spelling():
    joined = set(textnorm.tokens('می‌خوام برم باشگاه'))
    assert 'میخوام' in joined
    assert set(textnorm.tokens('می خوام برم باشگاه')) <= joined


def test_character_ngrams_share_a_stem_across_affixes():
    assert set(textnorm.char_ngrams('میخواستم')) & set(textnorm.char_ngrams('نمیخواستم'))
    assert textnorm.char_ngrams('اب', 4) == ['اب']   # shorter than the window
    assert textnorm.char_ngrams('') == []


def test_run_on_speech_splits_into_sentences():
    assert len(textnorm.sentences('رفتم سر کار و بعدش پریا زنگ زد. خسته بودم')) >= 2
    assert len(textnorm.sentences('کجا رفتی؟ هیچ جا')) == 2
    assert textnorm.sentences('   ') == []


# --- embedders -------------------------------------------------------------

def test_ascii_hash_embedder_is_blind_to_farsi():
    """An `[a-z0-9]+` tokeniser embeds a Farsi diary to the zero vector, so
    retrieval over it is arbitrary — the finding that moved the brain's
    default embedder off `hash`."""
    vectors = embedding.make_embedder('ascii-hash').embed(['امروز با پریا دعوام شد'])
    assert not np.any(vectors)


def test_char_hash_prefers_a_paraphrase_over_an_unrelated_line():
    embedder = embedding.make_embedder('char-hash')
    vectors = embedder.embed(['دعوا با پریا سر کارهای خونه',
                              'باز با پریا دعوا کردیم سر خونه',
                              'نامه اداره مالیات رسید'])
    assert float(vectors[0] @ vectors[1]) > float(vectors[0] @ vectors[2])


def test_token_hash_is_normalised_and_nonzero_for_farsi():
    vectors = embedding.make_embedder('token-hash').embed(['خواب بی‌خوابی کمردرد'])
    assert np.any(vectors)
    assert abs(float(np.linalg.norm(vectors[0])) - 1.0) < 1e-5


# --- chunking --------------------------------------------------------------

@pytest.mark.parametrize('chunker', ('message', 'turn-pair', 'semantic-drift'))
def test_message_preserving_chunkers_cover_every_turn(session, chunker):
    """No message may be dropped: the ground truth cites evidence by message
    index."""
    cfg = IndexConfig(chunker=chunker, embedder='char-hash', contextual=False)
    chunks = chunking.chunk_session(session, cfg, embedding.make_embedder('char-hash'))
    covered = set()
    for chunk in chunks:
        covered.update(range(chunk.msg_start, chunk.msg_end + 1))
    assert covered == set(range(len(session['messages'])))


def test_every_chunker_produces_unique_ids_and_nonempty_text(session):
    embedder = embedding.make_embedder('char-hash')
    for chunker in ('fixed', 'fixed-overlap', 'message', 'turn-pair', 'session',
                    'semantic-drift'):
        cfg = IndexConfig(chunker=chunker, embedder='char-hash')
        chunks = chunking.chunk_session(session, cfg, embedder)
        assert chunks, chunker
        assert len({c.id for c in chunks}) == len(chunks), chunker
        assert all(c.text.strip() for c in chunks), chunker


def test_fixed_chunker_matches_the_production_packing(session):
    """Same greedy 500-char packing the brain ships, or the comparison is
    against a straw man."""
    from raglab.chunking import chunk_text
    cfg = IndexConfig(chunker='fixed', chunk_chars=500, contextual=False)
    ours = chunking.chunk_session(session, cfg, embedding.make_embedder('char-hash'))
    theirs = chunk_text(corpus.session_text(session), 500)
    assert [c.text for c in ours] == theirs


def test_contextual_prefix_situates_the_chunk(session):
    cfg = IndexConfig(chunker='message', contextual=True)
    chunk = chunking.chunk_session(session, cfg, embedding.make_embedder('char-hash'))[0]
    assert session['date'] in chunk.prefix
    assert session['mood']['label'] in chunk.prefix
    assert chunk.body and not chunk.body.startswith('[')


def test_overlap_chunker_repeats_material_between_windows(session):
    cfg = IndexConfig(chunker='fixed-overlap', chunk_chars=300, overlap=150,
                      contextual=False)
    chunks = chunking.chunk_session(session, cfg, embedding.make_embedder('char-hash'))
    if len(chunks) < 2:
        pytest.skip('session too short to window')
    total = sum(len(c.text) for c in chunks)
    assert total > len(corpus.session_text(session))


def test_semantic_drift_cuts_at_an_explicit_topic_shift():
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
    cfg = IndexConfig(chunker='message')
    chunk = chunking.chunk_session(session, cfg, embedding.make_embedder('char-hash'))[0]
    for key, value in chunk.metadata().items():
        assert isinstance(value, (str, int, float, bool)), key


def test_importance_rises_with_emotional_intensity():
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


def test_the_corpus_declares_its_habits_the_way_a_board_card_does(diary):
    habits = diary['habits']
    assert habits, 'the corpus must carry the habits the diarist tracks'
    for slug, habit in habits.items():
        assert habit['freq'] in HABIT_FREQS, slug
        assert habit['count'] >= 1, slug
        assert habit['title_fa'], slug
        assert isinstance(habit['times'], list), slug
        assert isinstance(habit['history'], dict), slug
    # All four board cadences are not required, but a corpus that only ever
    # measured weekly habits would leave the monthly and daily period arithmetic
    # untested by every run.
    assert {h['freq'] for h in habits.values()} >= {'daily', 'weekly', 'monthly'}


def test_every_habit_completion_sits_in_the_period_it_is_filed_under(diary):
    """`habitHistory` is bucketed by period id, so a date filed under the
    wrong bucket would make every count wrong."""
    for slug, habit in diary['habits'].items():
        for period, days in habit['history'].items():
            for day in days:
                assert habit_period(habit['freq'], day) == period, f'{slug} {day}'


def test_a_habit_is_never_punched_more_often_than_its_period_asks(diary):
    """The punch strip has exactly `count` boxes, so more completions than
    that in one period could not have come from the board."""
    for slug, habit in diary['habits'].items():
        for period, days in habit['history'].items():
            assert len(days) <= habit['count'], f'{slug} {period}'
            assert len(days) == len(set(days)), f'{slug} {period} has a repeat'


def test_the_habit_sessions_joined_the_corpus_without_disturbing_it(diary):
    """Additive on purpose: appended on dates the corpus had not used, so
    every pre-existing session stays exactly as it was."""
    sessions = diary['sessions']
    ids = [s['session_id'] for s in sessions]
    assert len(ids) == len(set(ids)), 'a session id was reused'
    assert ids == sorted(ids), 'the corpus must stay chronological'
    habit_sessions = [s for s in sessions if 'habit-tracking' in s['recurring_threads']]
    assert len(habit_sessions) >= 8
    period = diary['meta']['period']
    for s in habit_sessions:
        assert period['from'] <= s['date'] <= period['to'], s['session_id']


def test_the_habit_storyline_is_described_like_every_other_thread(diary):
    """thread_layer builds its digest title from this description; a thread with
    no entry gets an empty one, which reads as a bug in the digest."""
    assert diary['threads']['habit-tracking']


def test_every_chunk_reports_a_habit_field_even_when_it_has_none(session):
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
    scope = query.resolve_time_scope(question, '2026-07-28')
    assert scope is not None, question
    assert (scope.from_int, scope.to_int) == (expect_from, expect_to)


def test_untimed_question_has_no_scope():
    assert query.resolve_time_scope('چرا با پریا دعوا می‌کنیم؟', '2026-07-28') is None


def test_relative_month_scope_is_the_previous_calendar_month():
    scope = query.resolve_time_scope('ماه پیش چی کار کردم؟', '2026-07-28')
    assert scope and (scope.from_int, scope.to_int) == (20260601, 20260630)


def test_where_clause_overlaps_rather_than_contains():
    """A chunk whose span straddles the edge of the window is kept: a scope
    asks about a period, not that the evidence sit entirely inside it."""
    scope = query.TimeScope(20260101, 20260131, 'دی', 'jalali-month')
    clause = query.where_clause(scope)
    assert clause['$and'][0] == {'span_from': {'$lte': 20260131}}
    assert clause['$and'][1] == {'span_to': {'$gte': 20260101}}
    assert query.where_clause(None) is None


def test_expansion_adds_a_synonym_variant():
    variants = query.expand('دعوا با همسرم سر چی بود؟')
    assert len(variants) >= 2
    assert any('پریا' in v for v in variants)


def test_keyword_query_strips_interrogatives():
    assert 'چی' not in query.keyword_query('حال مامان چی شد؟')


# --- retrieval primitives --------------------------------------------------

def test_bm25_finds_the_document_with_the_rare_term():
    bm25 = retrieval.BM25(['نامه اداره مالیات رسید و جریمه خوردم',
                           'با پریا دعوا کردیم', 'رفتم پیاده‌روی'])
    top = bm25.top('مالیات جریمه', 2)
    assert top and top[0][0] == 0


def test_bm25_respects_the_allowed_mask():
    bm25 = retrieval.BM25(['مالیات', 'مالیات'])
    allowed = np.array([False, True])
    assert [i for i, _ in bm25.top('مالیات', 2, allowed)] == [1]


def test_rrf_ranks_a_document_both_retrievers_agree_on_first():
    fused = retrieval.rrf([['a', 'b', 'c'], ['b', 'a', 'd']])
    assert max(fused, key=fused.get) in ('a', 'b')
    assert fused['a'] > fused['c'] and fused['b'] > fused['d']


def test_mmr_breaks_up_near_duplicates():
    vectors = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    relevance = np.array([1.0, 0.99, 0.5], dtype=np.float32)
    assert retrieval.mmr(vectors, relevance, 2, 1.0) == [0, 1]
    assert retrieval.mmr(vectors, relevance, 2, 0.5) == [0, 2]


def test_mmr_falls_back_when_vectors_are_missing():
    relevance = np.array([0.2, 0.9], dtype=np.float32)
    assert retrieval.mmr(np.zeros((0, 2), dtype=np.float32), relevance, 2, 0.5) == [1, 0]


def test_recency_weight_halves_after_one_half_life():
    weight = retrieval.recency_weight(20260101, 20260701, 180.0)
    assert 0.4 < weight < 0.6


def test_llm_grade_parser_defaults_unscored_lines_to_neutral():
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
    retrieved, gold = ['a', 'x', 'b'], ['a', 'b', 'c']
    assert metrics.recall_at_k(retrieved, gold, 3) == pytest.approx(2 / 3)
    assert metrics.precision_at_k(retrieved, gold, 3) == pytest.approx(2 / 3)
    assert metrics.mrr(retrieved, gold) == 1.0
    assert metrics.hit_at_k(['x'], gold, 1) == 0.0
    assert metrics.ndcg_at_k(['a', 'b'], gold, 2) > metrics.ndcg_at_k(['x', 'a'], gold, 2)


def test_quote_recall_needs_the_answering_sentence_not_just_the_session():
    question = {'evidence': [{'session_id': 's1', 'message_indices': [0],
                              'quote': 'آذر تموم شد و از هیچ شرکتی هیچ خبری نیس'}]}
    assert metrics.quote_recall('حرف‌های دیگری از همان نشست', question) == 0.0
    assert metrics.quote_recall('گفتم آذر تموم شد و از هیچ شرکتی هیچ خبری نیس بعدش',
                                question) == 1.0


def test_quote_recall_tolerates_whitespace_normalisation():
    question = {'evidence': [{'quote': 'می خوام برم باشگاه', 'session_id': 's',
                              'message_indices': [0]}]}
    assert metrics.quote_recall('گفت می  خوام   برم باشگاه', question) == 1.0


def test_latest_state_session_is_the_newest_evidence():
    question = {'evidence': [{'session_id': '2025-12-01-a'},
                             {'session_id': '2026-05-12-a'}]}
    assert metrics.latest_state_session(question) == '2026-05-12-a'


def test_aggregate_reports_per_type_and_a_headline():
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


def test_an_answerer_that_could_not_be_reached_says_so_on_the_row(ground_truth):
    """`pipeline._llm_answer` catches everything the model raises and returns
    the canonical refusal, so a CliError or a timeout looks exactly like "the
    diary is silent about that" unless something on the row says otherwise."""
    class Unreachable:
        def invoke(self, messages, **kwargs):
            raise clichat.CliError('claude did not answer within 600s')

    question = next(q for q in ground_truth['questions'] if q['answerable'])
    outcome = pipeline.Outcome(question=question['question_fa'], contexts=[])
    outcome.answer = pipeline._llm_answer(outcome, Unreachable(), 'sonnet')
    row = metrics.score_question(question, outcome, k=5)
    assert row['answer'] == pipeline.REFUSAL
    assert 'did not answer' in row['answer_error']

    # And a run where the model did answer carries no such field, so its presence
    # means one thing.
    answered = pipeline.Outcome(question=question['question_fa'], contexts=[],
                                answer='یک جواب')
    assert 'answer_error' not in metrics.score_question(question, answered, k=5)
