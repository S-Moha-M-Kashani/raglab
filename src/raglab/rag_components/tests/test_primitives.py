"""Text normalisation, embedders, chunking, habits, query understanding,
retrieval primitives and metrics — the lab's building blocks, tested in
isolation from any index or pipeline."""
import math

import numpy as np
import pytest

from raglab.rag_components.indexing import chunking_strategies as chunking
from raglab.llm_backends import cli_subprocess_chat as clichat
from raglab.corpora import corpus_reading as corpus
from raglab.rag_components.indexing import embedding_backends as embedding
from raglab.evaluation import deterministic_metrics as metrics
from raglab.rag_components import question_to_answer_pipeline as pipeline
from raglab.rag_components.retrieval import query_understanding as query
from raglab.rag_components.retrieval import (
    retrieve_fuse_rerank_grade as retrieval)
from raglab.rag_components.retrieval import farsi_text_normalizer as textnorm
from raglab.configuration import split_plan
from raglab.configuration.lab_config import IndexConfig


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


# --- the split plan's two forms -------------------------------------------

@pytest.mark.parametrize('line', [
    'document',
    'document / part',
    'document / role=user',
    'document / drift',
    'document / "\\n\\n" / "\\n" / ". "',
    'document / "\\n\\n" and role=assistant',
    'document / drift or "ولش کن" or "بگذریم" / part over-budget',
])
def test_every_plan_round_trips_between_its_typed_and_stored_forms(line):
    # this is a unit test
    """The stored list is what the fingerprint hashes; the line is what a
    person types into a sweep candidate or reads on a knob page. Each must
    come back from the other unchanged, and two spellings differing only in
    whitespace are one plan."""
    stages = split_plan.parse(line)
    assert split_plan.text(stages) == line
    assert split_plan.parse(split_plan.text(stages)) == stages
    assert split_plan.parse(line.replace(' / ', '/').replace(' or ', '  or  ')) == stages


@pytest.mark.parametrize('line, names', [
    ('part / document', 'must begin with the document stage'),
    ('document / "\\n" / drift', 'drift stage cannot follow a separator'),
    ('document / ". " / role=user', 'label stage cannot follow a separator'),
    ('document / speaker=chair', "declares no part-level label 'speaker'"),
    ('document / role=coach', "'role' takes one of user, assistant, not 'coach'"),
])
def test_an_impossible_plan_is_refused_by_name(line, names):
    # this is a unit test
    """A plan that cannot do what it says is refused before a build, and the
    refusal names what was wrong — the stage that needs parts after the one
    that cut them away, or the label the selected corpus never declared."""
    declared = {'role': {'applies_to': ['part', 'chunk'], 'values': ['user', 'assistant']}}
    problems = split_plan.problems(split_plan.parse(line), declared)
    assert any(names in problem for problem in problems), problems


def test_a_stage_takes_one_combinator_and_never_both():
    # this is a unit test
    with pytest.raises(ValueError, match='not both'):
        split_plan.parse('document / "\\n\\n" or "\\n" and role=user')
    assert split_plan.problems(({'kind': 'document'},
                                {'kind': 'separator', 'atoms': ({'text': 'x'},),
                                 'join': 'xor', 'when': 'always'}), {}) == [
        'split_plan stage 1: a stage combines its atoms with "or" or with '
        '"and", not \'xor\'']


# --- the normalisers ----------------------------------------------------------

def test_the_declared_language_picks_the_normaliser_and_a_name_overrides_it():
    # this is a unit test
    """A corpus declaring `fa` gets the Persian folds it always had; one
    declaring `de` is not folded by another language's rules — the digit `٣`
    stays a Farsi digit under the neutral normaliser and becomes `3` under the
    Persian one. Either may be named for any corpus, and an unknown name is
    refused rather than replaced."""
    from raglab.rag_components.retrieval import text_normalizers
    assert text_normalizers.resolve('', 'fa') is textnorm
    assert text_normalizers.resolve('', 'de') is text_normalizers.NEUTRAL
    assert text_normalizers.resolve('persian', 'de') is textnorm
    assert text_normalizers.NEUTRAL.tokens('Sitzung ٣٣ heute') == ['sitzung', '٣٣', 'heute']
    assert textnorm.tokens('Sitzung ٣٣ heute') == ['sitzung', '33', 'heute']
    with pytest.raises(ValueError, match='unknown normalizer'):
        text_normalizers.resolve('klingon', 'de')
    from raglab.configuration.lab_config import LabConfig
    assert any('unknown normalizer' in problem for problem in
               LabConfig.from_dict({'index': {'normalizer': 'klingon'}}).validate())


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

def _plan(line: str, **knobs) -> IndexConfig:
    return IndexConfig(split_plan=split_plan.parse(line), embedder='char-hash',
                       contextual=False, **knobs)


def _cut(document, line, embedder=None, label_fields=None, language='fa', **knobs):
    return chunking.chunk_document(
        document, _plan(line, **knobs), embedder or embedding.make_embedder('char-hash'),
        label_fields, language)


# The six chunkers the lab used to name, as the six plans that replaced
# them. The budget is set past every fixture document so nothing is divided
# — a plan cuts where its stages say and the budget closes it, so `session`
# is `document` only while the document fits.
OLD_CHUNKERS = {
    'session': ('document', 100_000, 0),
    'message': ('document / part', 100_000, 0),
    'turn-pair': ('document / role=user', 100_000, 0),
    'semantic-drift': ('document / drift', 100_000, 0),
    'fixed': ('document', 500, 0),
    'fixed-overlap': ('document', 500, 100),
}


@pytest.mark.parametrize('chunker', list(OLD_CHUNKERS))
def test_every_old_chunker_is_a_plan_that_yields_unique_nonempty_chunks_over_every_part(
        document, label_fields, language, chunker):
    # this is a unit test
    """The plan for each retired chunker must produce unique ids over nonempty
    text, and the structural plans compute `part_start`/`part_end` from where
    each chunk actually begins and ends, so no part may be dropped from that
    span — the ground truth cites evidence by part."""
    line, budget, overlap = OLD_CHUNKERS[chunker]
    chunks = _cut(document, line, label_fields=label_fields, language=language,
                  chunk_chars=budget, overlap=overlap)
    assert chunks, chunker
    assert len({c.id for c in chunks}) == len(chunks), chunker
    assert all(c.text.strip() for c in chunks), chunker
    parts = document['document_content']
    covered = set()
    for chunk in chunks:
        if chunk.part_start >= 0:
            covered.update(range(chunk.part_start, chunk.part_end + 1))
    if chunker in ('fixed', 'fixed-overlap'):
        for part in parts:
            assert part['text'].split()[0] in ' '.join(c.text for c in chunks)
    else:
        assert covered == set(range(len(parts))), chunker


def test_the_six_plans_reproduce_the_six_old_chunkers_byte_for_byte(
        document, label_fields, language):
    # this is a unit test
    """The old output, captured on this same fixture document before the
    rewrite and frozen here as its own shape rather than recomputed: the
    part spans of each chunk and the text each carried. `part_prefix='role'`
    puts back the `role: ` the old readers always wrote, and the embedder
    tokenises with the Persian normaliser the old code applied to every
    corpus, so the drift signal is the same signal. The two length-based
    chunkers recorded no span at all; the plan knows a whole document it
    never divided spans every part, so for them the text alone is compared."""
    embedder = embedding.make_embedder('char-hash', normalizer=textnorm)
    expected = {
        'session': [(0, 5)],
        'message': [(i, i) for i in range(6)],
        'turn-pair': [(0, 1), (2, 3), (4, 5)],
        'semantic-drift': [(0, 1), (2, 2), (3, 5)],
    }
    for chunker, (line, budget, overlap) in OLD_CHUNKERS.items():
        chunks = _cut(document, line, embedder, label_fields, language,
                      chunk_chars=budget, overlap=overlap, part_prefix='role')
        texts = [c.text for c in chunks]
        if chunker in expected:
            assert [(c.part_start, c.part_end) for c in chunks] == expected[chunker], chunker
            joined = [corpus.document_text(
                {'document_content': document['document_content'][a:b + 1]},
                prefix='role') for a, b in expected[chunker]]
            assert texts == joined, chunker
            for chunk in chunks:
                assert chunk.text.startswith(('user: ', 'assistant: ')), chunker
        else:
            whole = corpus.document_text(document, prefix='role')
            packed = (chunking._windows(whole, 500, 100) if overlap
                      else chunking.chunk_text(whole, 500))
            assert texts == packed, chunker


def test_a_label_boundary_opens_a_turn_without_naming_the_other_speaker():
    # this is a unit test
    """`role=user` reproduces `turn-pair` — a chunk begins at each user part
    and runs to the part before the next — and, because it names only the
    opener, works on a corpus with three declared speakers and on a turn the
    old pairing could not express (one question, two replies)."""
    fields = {'speaker': {'type': 'string', 'applies_to': ['part', 'chunk'],
                          'values': ['chair', 'member', 'guest']}}
    meeting = {'corpus_document_id': 7, 'document_content': [
        {'text': 'Opening the budget item.', 'labels': {'speaker': 'chair'}},
        {'text': 'We are over by ten percent.', 'labels': {'speaker': 'member'}},
        {'text': 'The vendor quote arrived late.', 'labels': {'speaker': 'guest'}},
        {'text': 'Next item, the schedule.', 'labels': {'speaker': 'chair'}},
        {'text': 'Two weeks behind.', 'labels': {'speaker': 'member'}}]}
    chunks = _cut(meeting, 'document / speaker=chair', label_fields=fields,
                  language='en', chunk_chars=10_000)
    assert [(c.part_start, c.part_end) for c in chunks] == [(0, 2), (3, 4)]
    # every part-level label of the parts a chunk spans reaches it
    assert chunks[0].labels['speaker'] == ['chair', 'member', 'guest']
    # and no speaker prefix is written into the text unless asked for by name
    assert chunks[0].text.startswith('Opening the budget item.')
    prefixed = _cut(meeting, 'document / speaker=chair', label_fields=fields,
                    language='en', chunk_chars=10_000, part_prefix='speaker')
    assert prefixed[0].text.startswith('chair: Opening the budget item.\nmember: ')


# Three paragraphs, the middle one over any budget the tests use, its
# sentences under it — the shape a coarse-to-fine list of separators is for.
DELIMITED = ('Short opening paragraph.\n\n'
             'A second paragraph too long for the budget. '
             'It runs on with several sentences. One more here.\n\n'
             'Tail.')


def _document_of(text: str) -> dict:
    return {'corpus_document_id': 1, 'document_content': [{'text': text}]}


def test_successive_separator_stages_descend_and_stop_once_a_piece_fits():
    # this is a unit test
    """Coarse-to-fine as stages: a paragraph that fits the budget is left
    whole by the sentence stage, and a paragraph that does not is cut at its
    sentences. A separator applies only over budget by default, so the
    document under a large budget is not cut at all."""
    chunks = _cut(_document_of(DELIMITED), 'document / "\\n\\n" / ". "',
                  language='en', chunk_chars=45)
    assert [c.text for c in chunks] == [
        'Short opening paragraph.',
        'A second paragraph too long for the budget',
        'It runs on with several sentences',
        'One more here.',
        'Tail.']
    assert all(c.part_start == -1 for c in chunks), 'a separator knows no part span'
    whole = _cut(_document_of(DELIMITED), 'document / "\\n\\n" / ". "',
                 language='en', chunk_chars=1000)
    assert [c.text for c in whole] == [DELIMITED]


def test_a_stage_set_to_always_cuts_a_piece_that_already_fits():
    # this is a unit test
    """The per-stage toggle in both directions: `always` on a separator cuts
    the short document, and `over-budget` on a part stage leaves a short
    document whole while a long one is still cut into its parts."""
    always = _cut(_document_of(DELIMITED), 'document / "\\n\\n" always',
                  language='en', chunk_chars=1000)
    assert len(always) == 3
    two_parts = {'corpus_document_id': 1, 'document_content': [
        {'text': 'one'}, {'text': 'two'}]}
    assert len(_cut(two_parts, 'document / part over-budget', chunk_chars=1000)) == 1
    assert len(_cut(two_parts, 'document / part', chunk_chars=1000)) == 2


def test_or_cuts_at_any_atom_and_and_only_where_a_literal_and_a_label_both_hold():
    # this is a unit test
    """The two combinators on one document. Under `or` every blank line and
    every single newline is a cut. Under `and` with a label boundary the
    blank line cuts only inside the part that label selects — the other
    part's blank line is left alone."""
    fields = {'role': {'type': 'string', 'applies_to': ['part', 'chunk'],
                       'values': ['user', 'assistant']}}
    document = {'corpus_document_id': 1, 'document_content': [
        {'text': 'first\n\nsecond\nthird', 'labels': {'role': 'user'}},
        {'text': 'fourth\n\nfifth', 'labels': {'role': 'assistant'}}]}
    either = _cut(document, 'document / "\\n\\n" or "\\n" always',
                  label_fields=fields, language='en', chunk_chars=1000)
    assert [c.text for c in either] == ['first', 'second', 'third', 'fourth', 'fifth']
    narrowed = _cut(document, 'document / "\\n\\n" and role=assistant always',
                    label_fields=fields, language='en', chunk_chars=1000)
    assert [c.text for c in narrowed] == ['first\n\nsecond\nthird\nfourth', 'fifth']


def test_word_packing_is_unchanged_as_the_budget_that_closes_every_plan():
    # this is a unit test
    """The greedy packing and the sliding window are the base case every
    plan ends in; their output is frozen here from before the plan existed."""
    text = 'Alpha beta.\nGamma  delta.\n\n\tEpsilon zeta eta.\n'
    assert chunking.chunk_text(text, 20) == [
        'Alpha beta. Gamma', 'delta. Epsilon zeta', 'eta.']
    assert chunking._windows(text, 20, 5) == [
        'Alpha beta. Gamma', 'Gamma delta. Epsilon', 'Epsilon zeta eta.']


def test_the_budget_divides_a_piece_no_stage_reduced_and_keeps_a_single_parts_span():
    # this is a unit test
    """After the last stage a piece still over the budget is divided at word
    boundaries. Cut from one part, the pieces still span that part — so its
    labels reach them; cut from a run of parts, they no longer know which."""
    fields = {'role': {'type': 'string', 'applies_to': ['part', 'chunk']}}
    document = {'corpus_document_id': 1, 'document_content': [
        {'text': ' '.join(f'w{i}' for i in range(40)), 'labels': {'role': 'user'}},
        {'text': 'short reply', 'labels': {'role': 'assistant'}}]}
    by_part = _cut(document, 'document / part', label_fields=fields,
                   language='en', chunk_chars=60)
    assert len(by_part) > 2
    assert {(c.part_start, c.part_end) for c in by_part[:-1]} == {(0, 0)}
    assert all(c.labels['role'] == 'user' for c in by_part[:-1])
    whole = _cut(document, 'document', label_fields=fields, language='en',
                 chunk_chars=60)
    assert len(whole) > 1 and all(c.part_start == -1 for c in whole)
    assert all('role' not in c.labels for c in whole)


def test_a_budget_in_model_units_refuses_an_embedder_that_cannot_count_them():
    # this is a unit test
    """`chunk_unit='tokens'` measures the budget in the embedder's own units;
    an embedder with no tokeniser is refused rather than counted in
    characters under the wrong name."""
    cfg = IndexConfig(split_plan=split_plan.parse('document'), chunk_unit='tokens',
                      embedder='sentence-transformers')
    with pytest.raises(ValueError, match='reports no model units'):
        chunking.budget_measure(cfg, embedding.make_embedder('char-hash'))

    class Counting:
        """A stand-in for a model-backed embedder whose tokeniser counts a word as two units."""
        class model:
            class tokenizer:
                @staticmethod
                def encode(word, add_special_tokens=False):
                    return [1, 2]
    measure = chunking.budget_measure(cfg, Counting())
    assert measure('three words here') == 6
    assert chunking.chunk_text('a b c d e f', 4, measure) == ['a b', 'c d', 'e f']


def test_contextual_prefix_situates_the_chunk():
    # this is a unit test
    document = {
        'corpus_document_id': 1,
        'document_content': [{'text': 'hello', 'labels': {'role': 'user'}},
                             {'text': 'hi', 'labels': {'role': 'assistant'}}],
        'document_metadata': {'recorded_at': '2026-01-01T00:00:00Z',
                              'mood_label': 'happy', 'topics': ['a', 'b']},
    }
    label_fields = {
        'recorded_at': {'type': 'string', 'format': 'date-time', 'description': 'x',
                        'applies_to': ['document', 'chunk']},
        'mood_label': {'type': 'string', 'description': 'x',
                       'applies_to': ['document', 'chunk']},
        'topics': {'type': 'array', 'items': {'type': 'string'}, 'description': 'x',
                  'applies_to': ['document', 'chunk']},
        'role': {'type': 'string', 'description': 'x', 'applies_to': ['part', 'chunk']},
    }
    cfg = IndexConfig(split_plan=split_plan.parse('document / part'), contextual=True)
    chunk = chunking.chunk_document(document, cfg, embedding.make_embedder('char-hash'),
                                    label_fields, 'en')[0]
    assert 'recorded_at: 2026-01-01T00:00:00Z' in chunk.prefix
    assert 'mood_label: happy' in chunk.prefix
    assert 'topics: a, b' in chunk.prefix
    assert chunk.body and not chunk.body.startswith('[')


def test_contextual_prefix_joins_list_values_by_the_corpus_language_comma():
    # this is a unit test
    """'the language's comma': derived from the corpus's own declared
    language, Farsi's «،» rather than the ASCII comma every other language
    (that declares no special one) gets."""
    document = {'corpus_document_id': 1, 'document_content': [{'text': 'سلام'}],
                'document_metadata': {'topics': ['كار', 'خونه']}}
    label_fields = {'topics': {'type': 'array', 'items': {'type': 'string'},
                               'description': 'x', 'applies_to': ['document', 'chunk']}}
    fa = chunking.contextual_prefix(document, label_fields, 'fa')
    en = chunking.contextual_prefix(document, label_fields, 'en')
    assert 'topics: كار، خونه' in fa
    assert 'topics: كار, خونه' in en


def test_contextual_prefix_never_shows_a_confidence_label():
    # this is a unit test
    """A label declaring `confidence_for` is a caveat on another label, never
    something to embed (schema `x-raglab-uses.confidences_are_caveats_not_signals`)."""
    document = {'corpus_document_id': 1, 'document_content': [{'text': 'x'}],
                'document_metadata': {'feeling': 'کلافه', 'feeling_confidence': 0.9}}
    label_fields = {
        'feeling': {'type': 'string', 'description': 'x',
                   'applies_to': ['document', 'chunk'], 'extracted': True},
        'feeling_confidence': {'type': 'number', 'description': 'x',
                               'applies_to': ['document', 'chunk'],
                               'confidence_for': 'feeling'},
    }
    prefix = chunking.contextual_prefix(document, label_fields, 'en')
    assert 'feeling: کلافه' in prefix
    assert 'feeling_confidence' not in prefix


def test_contextual_prefix_never_renders_a_nullable_labels_null_as_the_word_none():
    # this is a unit test
    """D4: a nullable label recorded `null` means "not recorded", never
    "recorded as nothing" — it must be skipped, not rendered as the literal
    string `'None'` (what plain `str(None)` would have written here)."""
    document = {'corpus_document_id': 1, 'document_content': [{'text': 'x'}],
                'document_metadata': {'feeling': None, 'topics': ['office']}}
    label_fields = {
        'feeling': {'type': 'string', 'nullable': True, 'description': 'x',
                   'applies_to': ['document', 'chunk']},
        'topics': {'type': 'array', 'items': {'type': 'string'},
                  'description': 'x', 'applies_to': ['document', 'chunk']},
    }
    prefix = chunking.contextual_prefix(document, label_fields, 'en')
    assert 'feeling' not in prefix
    assert 'None' not in prefix
    assert 'topics: office' in prefix


def test_contextual_prefix_drops_an_empty_item_inside_a_list_value():
    # this is a unit test
    """A list carrying a recorded-empty item alongside real ones renders
    only the real ones — `is_present` is the one rule both the per-value
    skip and the per-item join defer to, so an empty string never becomes a
    bare, confusing comma in the joined text."""
    document = {'corpus_document_id': 1, 'document_content': [{'text': 'x'}],
                'document_metadata': {'topics': ['office', '', 'deliveries']}}
    label_fields = {'topics': {'type': 'array', 'items': {'type': 'string'},
                               'description': 'x',
                               'applies_to': ['document', 'chunk']}}
    prefix = chunking.contextual_prefix(document, label_fields, 'en')
    assert 'topics: office, deliveries' in prefix


def test_is_present_treats_none_and_empty_string_as_absent_and_everything_else_as_present():
    # this is a unit test
    """The one presence rule shared by `contextual_prefix` and
    `summary_hierarchy_builder._metadata_groups` — pinned directly so the two
    functions can never silently drift apart about what "absent" means."""
    assert not chunking.is_present(None)
    assert not chunking.is_present('')
    assert chunking.is_present(0)
    assert chunking.is_present(False)
    assert chunking.is_present('x')
    assert chunking.is_present([])   # an empty list is a recorded value, not absence


def test_contextual_prefix_is_empty_without_any_document_level_label():
    # this is a unit test
    """D4: absence stays absence — no bracket at all when the document
    carries nothing declared at the document level, as smoke-mini's `role`
    (a part-level-only label) leaves it."""
    document = {'corpus_document_id': 1, 'document_content': [{'text': 'x'}]}
    label_fields = {'role': {'type': 'string', 'description': 'x',
                             'applies_to': ['part', 'chunk']}}
    assert chunking.contextual_prefix(document, label_fields, 'en') == ''


def _long_document() -> dict:
    """A synthetic document with enough text to guarantee at least two
    fixed-overlap windows at chunk_chars=300/overlap=150 — the real corpus's
    `document` fixture is not guaranteed to be long enough, which used to make
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
    return {'corpus_document_id': 1, 'document_content': [
        {'text': ' '.join(words[:midpoint]), 'labels': {'role': 'user'}},
        {'text': ' '.join(words[midpoint:]), 'labels': {'role': 'assistant'}}]}


def test_an_overlap_repeats_material_between_the_pieces_the_budget_makes():
    # this is a unit test
    """Adjacent pieces must share material, or the overlap is a number nothing
    reads. Run over a synthetic document built long enough to window at
    least twice, rather than skipping when the corpus document handed to it
    happens to be short — a test that can skip itself is a test that can
    assert nothing."""
    document = _long_document()
    chunks = _cut(document, 'document', chunk_chars=300, overlap=150)
    assert len(chunks) >= 2, 'the synthetic document must be long enough to window'
    total = sum(len(c.text) for c in chunks)
    assert total > len(corpus.document_text(document))
    # Not just longer overall: the tail of each window has to reappear,
    # verbatim, at the head of the next one — every word here is unique, so
    # this substring cannot be satisfied by chance.
    for a, b in zip(chunks, chunks[1:]):
        tail = ' '.join(a.text.split()[-3:])
        assert tail in b.text, (tail, b.text)


def test_the_drift_stage_cuts_at_a_marker_only_when_given_one():
    # this is a unit test
    """No language's phrases are compiled in: with no markers the stage cuts
    on similarity and its size ceiling alone, and the same corpus given the
    diarist's own topic-change phrase cuts there too."""
    # Three parts about the tax letter, the second opening with the diarist's
    # topic-change phrase, then one about a different subject entirely: the
    # similarity signal cuts before the last part whatever else is set, and
    # only the marker can cut before the second.
    fake = {'corpus_document_id': 1, 'document_content': [
        {'text': 'امروز کل روز درگیر مالیات بودم و نامه اداره مالیات'},
        {'text': 'حالا اینا رو ولش کن، نامه اداره مالیات و درگیر مالیات بودم'},
        {'text': 'نامه اداره مالیات، کل روز درگیر مالیات'},
        {'text': 'پریا سر کارهای خونه دوباره دعوا کرد، چه حسی داشتی'}]}
    embedder = embedding.make_embedder('char-hash', normalizer=textnorm)
    without = _cut(fake, 'document / drift', embedder, chunk_chars=500)
    assert [(c.part_start, c.part_end) for c in without] == [(0, 2), (3, 3)]
    with_marker = _cut(fake, 'document / drift or "ولش کن"', embedder, chunk_chars=500)
    assert [(c.part_start, c.part_end) for c in with_marker] == [(0, 0), (1, 2), (3, 3)]
    module = chunking.__file__
    assert 'ولش' not in open(module, encoding='utf-8').read(), (
        'no Persian phrase may be compiled into the splitter')


def test_chunk_metadata_is_chroma_safe(document, label_fields, language):
    # this is a unit test
    """Exercises the diary document's own keyed confidence label
    (`topics_confidence`, an object) alongside its lists and scalars — the
    kind of value `metadata()` has to flatten, not just the easy ones."""
    cfg = IndexConfig(split_plan=split_plan.parse('document / part'))
    chunk = chunking.chunk_document(document, cfg, embedding.make_embedder('char-hash'),
                                    label_fields, language)[0]
    for key, value in chunk.metadata().items():
        assert isinstance(value, (str, int, float, bool)), key


def test_importance_rises_with_emotional_intensity():
    # this is a unit test
    label_fields = {'mood_valence': {'type': 'number', 'minimum': 0, 'maximum': 10,
                                     'description': 'x', 'applies_to': ['document'],
                                     'ranks': True}}
    calm = {'document_metadata': {'mood_valence': 2}}
    wrecked = {'document_metadata': {'mood_valence': 9}}
    assert (chunking.importance_of(wrecked, label_fields)
            > chunking.importance_of(calm, label_fields))


def test_importance_is_zero_without_a_declared_ranks_label():
    # this is a unit test
    """D6: importance needs a declared source or it is zero — never a
    neutral default invented in its place."""
    document = {'document_metadata': {'mood_valence': 9}}
    assert chunking.importance_of(document, {}) == 0.0


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
    question = {'relevant_corpus_documents': [{'corpus_document_id': 1, 'evidence': [
        {'text': 'آذر تموم شد و از هیچ شرکتی هیچ خبری نیس', 'fidelity': 'verbatim'}]}]}
    assert metrics.quote_recall(answer, question) == expected


def test_quote_recall_ignores_a_paraphrase_or_a_computed_entry():
    # this is a unit test
    """A lexical match against a non-verbatim entry measures nothing — the
    schema change from `evidence[].quote` to `evidence[].fidelity`, and the
    real behaviour change task-5 introduces: only `fidelity: verbatim` may be
    scored lexically (replaces the deleted
    test_latest_state_session_is_the_newest_evidence, which pinned a metric
    tied to the now-deleted fixed `type` vocabulary — see aggregate() below)."""
    question = {'relevant_corpus_documents': [{'corpus_document_id': 1, 'evidence': [
        {'text': 'یک جمله که در متن نیست', 'fidelity': 'computed'}]}]}
    # nothing to lexically match against a computed-only question: undefined
    # (nan), not zero — even when the context text contains that exact string.
    assert math.isnan(metrics.quote_recall('یک جمله که در متن نیست', question))


def test_aggregate_reports_overall_means_and_a_headline():
    # this is a unit test
    """`by_type`/`by_difficulty` are dropped here (not merely renamed): `type`
    and `difficulty` are no longer guaranteed fields on a row — a corpus
    declares whatever question labels it likes (D7) — so a fixed-vocabulary
    breakdown cannot be substituted, only removed. `aggregate()`'s own
    docstring explains what replaces it (nothing at this layer;
    `selection_note`'s `by_<balance>` reports the run's own breakdown)."""
    rows = [
        {'id': 'q1', 'behavior': 'answer',
         'recall': 1.0, 'quote_recall': 1.0, 'ndcg': 1.0, 'hit': 1.0,
         'layers': ['chunk'], 'latency_ms': 5},
        {'id': 'q2', 'behavior': 'abstain',
         'abstained_correctly': 1.0, 'layers': [], 'latency_ms': 5},
    ]
    summary = metrics.aggregate(rows)
    assert summary['n_questions'] == 2
    assert 'by_type' not in summary and 'by_difficulty' not in summary
    assert summary['overall']['recall'] == 1.0
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


def test_llm_answer_prompt_requires_the_question_language():
    # this is a unit test
    """The prompt must tell the model to answer in the question's language."""
    class Answered:
        content = 'answer'

    class RecordingModel:
        def __init__(self):
            self.messages = None

        def invoke(self, messages, **kwargs):
            self.messages = messages
            return Answered()

    for question, language in (
            ('امروز چه خبر بود؟', 'Persian'),
            ('What happened today?', 'English'),
            ('Was ist heute passiert?', 'German')):
        model = RecordingModel()
        outcome = pipeline.Outcome(
            question=question,
            contexts=[pipeline.Context('chunk-1', 'Evidence text.', 'session-1',
                                       '2026-01-01', 1.0)],
        )
        assert pipeline._llm_answer(outcome, model, 'answerer') == 'answer'
        assert f'answer in {language}' in model.messages[0]['content']
