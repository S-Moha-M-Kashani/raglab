"""Screening a judge before it is allowed to grade — a held-out task with
known answers, run before any judge is trusted with the leaderboard."""
import re
from dataclasses import replace

import pytest

from raglab import corpus, judgescreen, models, textnorm
from raglab.config import GenerationConfig, LabConfig

from conftest import LAB_SETTINGS


# --- screening a judge before it is allowed to grade ------------------------
# A weak judge does not produce noisy rankings — it produces confident wrong
# ones, so the judge is screened before it is trusted to grade anything.

def test_the_screen_pairs_a_verified_answer_with_one_fabricated_number(
        diary, ground_truth):
    """Built from the ground truth, not hand-authored: a supported claim is a
    question's own verified answer, and its partner is that answer with one
    numeral changed to one the context never states."""
    sessions = corpus.sessions_by_id(diary)
    items = judgescreen.build_items(ground_truth, sessions, pairs=4)
    assert len(items) == 8
    yes = [i for i in items if i.supported]
    no = [i for i in items if not i.supported]
    assert len(yes) == len(no) == 4, 'an unbalanced screen flatters a constant judge'
    for supported, fabricated in zip(yes, no):
        assert supported.question_id == fabricated.question_id
        assert supported.claim != fabricated.claim
        # Word-for-word identical apart from digits: that is what removes the
        # lexical shortcut.
        strip = lambda text: ''.join(c for c in text if not c.isdigit()
                                     and c not in '۰۱۲۳۴۵۶۷۸۹')
        assert strip(supported.claim) == strip(fabricated.claim)


def test_the_screen_measures_how_much_word_overlap_could_explain(diary,
                                                                ground_truth):
    """Reported, not assumed — and it is not zero, which is a deliberate
    trade: correct labels matter more than a screen a word-counter could
    partly game. The check that actually decides is degeneracy, which no
    lexical shortcut can pass."""
    items = judgescreen.build_items(ground_truth, corpus.sessions_by_id(diary),
                                    pairs=6)
    signal = judgescreen.lexical_signal(items)
    assert signal['difference'] is not None
    assert 'blind' in signal
    # Small enough that overlap cannot be the whole story: the fabricated claims
    # still share almost all their vocabulary with the context.
    assert abs(signal['difference']) <= 0.15, signal


def test_the_screen_dates_its_context_the_way_the_pipeline_does(diary,
                                                               ground_truth):
    """Diary messages are spoken and almost never state a date; the date is
    session metadata, so a judge shown bare message text would refuse a
    true claim *for the right reason*. The pipeline under test has the same
    problem and solves it the same way (`IndexConfig.contextual`)."""
    sessions = corpus.sessions_by_id(diary)
    items = judgescreen.build_items(ground_truth, sessions, pairs=6)
    for item in items:
        for line in item.context:
            assert re.match(r'^\[\d{4}-\d{2}-\d{2}\]', line), line


def test_a_screened_claim_is_one_sentence_not_a_whole_answer(diary,
                                                            ground_truth):
    """A reference answer spans several clauses and sessions, so a judge
    asked to entail all of it against one evidence set is right to refuse.
    RAGAS's own faithfulness decomposes a response into atomic statements
    before judging, so an undecomposed paragraph would not resemble the
    real task either."""
    sessions = corpus.sessions_by_id(diary)
    items = judgescreen.build_items(ground_truth, sessions, pairs=6)
    answers = {q['id']: q['answer_fa'] for q in ground_truth['questions']}
    for item in items:
        # One sentence. Not "shorter than the answer": a single-sentence answer
        # legitimately yields a claim of the same length, and asserting length
        # would be testing the fixture's prose rather than the decomposition.
        assert len(textnorm.sentences(item.claim)) == 1, item.id
        assert len(item.claim) <= len(answers[item.question_id]), item.id
        # And it is anchored: it states a number the context also states, so the
        # context can actually settle it either way.
        anchored = [n for n in judgescreen.NUMERAL.findall(item.claim)
                    if textnorm.normalize(n)
                    in textnorm.normalize(' '.join(item.context))]
        assert anchored or not item.supported, item.id


def test_the_fabricated_number_is_one_the_context_never_states(diary,
                                                              ground_truth):
    """Otherwise the claim is labelled unsupported while being arguably
    supported, and the screen would disqualify the judge that got it right."""
    sessions = corpus.sessions_by_id(diary)
    items = judgescreen.build_items(ground_truth, sessions, pairs=6)
    for item in (i for i in items if not i.supported):
        context = textnorm.normalize(' '.join(item.context))
        original = next(i for i in items
                        if i.question_id == item.question_id and i.supported)
        changed = [n for n in judgescreen.NUMERAL.findall(item.claim)
                   if n not in judgescreen.NUMERAL.findall(original.claim)]
        assert changed, item.id
        for numeral in changed:
            assert textnorm.normalize(numeral) not in context, (item.id, numeral)


def test_a_question_that_cannot_be_mutated_cleanly_is_skipped(diary):
    """No mutation is better than a mislabelled one."""
    sessions = corpus.sessions_by_id(diary)
    # No numerals at all, so nothing can be fabricated.
    ground_truth = {'questions': [
        {'id': 'q-x', 'answerable': True, 'answer_fa': 'هیچ عددی اینجا نیست',
         'evidence': [{'quote': 'متن بدون عدد', 'session_id': 'nope',
                       'message_indices': []}]}]}
    assert judgescreen.build_items(ground_truth, sessions, pairs=4) == []


def test_the_screen_reads_a_ragas_shaped_reply_and_nothing_looser():
    """RAGAS asks for nested JSON and retries on malformed output, so a model that
    judges well but writes prose spends its speed advantage on retries. Counting
    a bare 'yes' as an answer here would hide exactly that cost."""
    good = '{"statements": [{"statement": "x", "verdict": 1, "reason": "y"}]}'
    assert judgescreen._verdict(good) == 1
    # A fenced block is a formatting habit, not a failure to answer.
    assert judgescreen._verdict('```json\n{"statements":[{"verdict":0}]}\n```') == 0
    assert judgescreen._verdict('Yes, it is supported.') is None
    assert judgescreen._verdict('{"statements": []}') is None
    assert judgescreen._verdict('{"verdict": 1}') is None
    assert judgescreen._verdict('') is None


def test_a_constant_judge_is_reported_as_degenerate_not_as_fifty_percent():
    """The field that decides. A model answering the same way every time is
    unusable at any accuracy, because it cannot separate two candidates — and on
    a balanced set it posts 0.5, which reads like a merely weak judge."""
    from raglab.judgescreen import Call, score
    calls = [Call(item_id=f'i{i}', supported=i % 2 == 0, verdict=1, parsed=True,
                  seconds=1.0, prompt='p', reply='r') for i in range(8)]
    result = score(calls)
    assert result['degenerate'] is True
    assert result['accuracy'] == 0.5
    assert result['recall_supported'] == 1.0
    assert result['recall_unsupported'] == 0.0


def test_a_judge_that_tracks_the_claim_is_not_flagged_degenerate():
    from raglab.judgescreen import Call, score
    calls = [Call(item_id=f'i{i}', supported=i % 2 == 0,
                  verdict=int(i % 2 == 0), parsed=True, seconds=1.0,
                  prompt='p', reply='r') for i in range(8)]
    result = score(calls)
    assert result['degenerate'] is False and result['accuracy'] == 1.0


def test_unparseable_replies_are_counted_separately_from_wrong_ones():
    """Two different problems with two different fixes: a prompt/format issue and
    a comprehension issue. Folding them together would send you tuning the wrong
    one."""
    from raglab.judgescreen import Call, score
    calls = [Call(item_id='a', supported=True, verdict=1, parsed=True,
                  seconds=1.0, prompt='p', reply='r'),
             Call(item_id='b', supported=False, verdict=None, parsed=False,
                  seconds=1.0, prompt='p', reply='I think maybe')]
    result = score(calls)
    assert result['schema_failures'] == 1
    assert result['n_parsed'] == 1
    assert result['accuracy'] == 1.0, 'accuracy is over what could be graded'


def test_the_screen_refuses_to_run_without_a_backend(monkeypatch):
    """The same guard as the sweep, for the same reason: the fake provider judges
    every claim without failing, and a screen it passed would be a licence."""
    monkeypatch.setattr(judgescreen, 'load_lab_settings', lambda: LAB_SETTINGS)
    with pytest.raises(SystemExit, match='no LLM backend'):
        judgescreen.screen(['whatever:1b'], pairs=1)


def test_the_screen_keeps_every_prompt_and_reply_it_sent():
    """A screen that reported only an accuracy could not be re-read to see
    *how* a model failed — "it was a constant predictor" is a conclusion
    nobody can check from a bare number."""
    from dataclasses import fields

    from raglab.judgescreen import Call
    names = {f.name for f in fields(Call)}
    assert {'prompt', 'reply', 'verdict', 'parsed', 'seconds', 'usage'} <= names


def test_a_remote_slug_is_never_refused_on_the_strength_of_a_listing(monkeypatch):
    """OpenRouter's list is authoritative in one direction only: everything on it
    works, but a slug missing from it may still be valid — the routing suffixes
    (`:free`, `:floor`) do not appear as ids. Blocking runs that used to work is a
    worse failure than the mislabelled row this guard exists to prevent, so the
    refusal is scoped to the local backend, whose tag list *is* authoritative both
    ways."""
    keyed = replace(LAB_SETTINGS, openrouter_api_key='sk-x')
    monkeypatch.setattr(models, 'openrouter_ids',
                        lambda settings: frozenset({'openai/gpt-5-nano'}))
    cfg = LabConfig(generation=GenerationConfig(ragas_model='openai/gpt-5-mini:floor'))
    assert models.provider_problems(cfg, keyed) == []
