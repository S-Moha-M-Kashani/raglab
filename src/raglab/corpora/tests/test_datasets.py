"""One loader, one shape: the corpus/ground-truth contract every dataset
meets, and the rules that keep a score meaning something.

Every retrieval finding this lab has produced is a finding *about* the Farsi
diary fixture. Some are obviously general (an embedder that cannot represent
the script scores at chance) and some obviously are not (the Farsi
time-scope filter), and with one corpus there was no way to tell which was
which. So the tests here are mostly about not confusing the two: a dataset is
two files paired by id, an index built over one corpus can never answer a
question from another, and a dataset whose evidence does not hold is
refused rather than measured.

Deliberately out of scope here: the panel/inspector HTTP surface
(`panel_server.create_app`) and the chunking/retrieval pipeline
(`IndexRegistry`, `chunking_strategies`) still read the pre-refactor shape —
that is the next steps' work, not this seam's. Those surfaces get their own
tests once they speak the schema's vocabulary.
"""
import json

import pytest

from raglab.corpora import corpus_store as corpora
from raglab.corpora import dataset_import_contract as datasets
from raglab.evaluation import leaderboard

from raglab.configuration.lab_config import IndexConfig

BUNDLED = ('diary-en', 'diary-fa', 'support-en', 'meetings-de',
          'research-multihop', 'smoke-mini', 'nosrat-fa')
# The control corpora cover every failure mode; the diary is excluded
# from that specific check only because it is what every other check already
# exercises, not because it is special (D3 — it is an ordinary bundled pair).
# `nosrat-fa` is the control the set was missing: same language as the diary,
# a different domain and a document thirteen turns long, so a finding that
# survives the German meetings but not this one was about the diary's shape
# rather than about Farsi.
CONTROLS = ('support-en', 'meetings-de', 'research-multihop', 'nosrat-fa')


def _valid_pair(corpus_overrides: dict | None = None,
                ground_truth_overrides: dict | None = None) -> tuple[dict, dict]:
    """The smallest pair that passes: two documents, one question, one quote
    that is really in the document it cites."""
    quote = 'the roof was fixed on 3 March'
    corpus = {
        'corpus_dataset_metadata': {
            'dataset': 'tiny-test', 'name': 'Tiny', 'language': 'en'},
        'corpus_documents': [
            {'corpus_document_id': 1,
             'document_content': [
                 {'text': f'Good news — {quote}, at last.'}]},
            {'corpus_document_id': 2,
             'document_content': [{'text': 'Nothing to report.'}]},
        ],
    }
    ground_truth = {
        'groundtruth_dataset_metadata': {
            'name': 'Tiny — questions', 'corpus_ref': {'dataset': 'tiny-test'}},
        'groundtruth_dataset': [
            {'groundtruth_question_id': 1,
             'question': 'When was the roof fixed?',
             'expected_answer': {'behavior': 'answer', 'text': 'On 3 March.'},
             'relevant_corpus_documents': [
                 {'corpus_document_id': 1,
                  'evidence': [{'text': quote, 'fidelity': 'verbatim',
                                'part_labels': [{}]}]}]},
        ],
    }
    if corpus_overrides:
        corpus.update(corpus_overrides)
    if ground_truth_overrides:
        ground_truth.update(ground_truth_overrides)
    return corpus, ground_truth


@pytest.fixture
def imports_here(tmp_path, monkeypatch):
    monkeypatch.setenv('RAGLAB_DATASETS', str(tmp_path / 'datasets'))
    datasets.forget()
    yield tmp_path / 'datasets'
    datasets.forget()


# --- the contract ------------------------------------------------------------

def test_a_dataset_that_meets_the_contract_has_nothing_to_report():
    # this is a unit test
    corpus, ground_truth = _valid_pair()
    assert datasets.validate(corpus, ground_truth) == []


def test_a_dataset_with_a_quote_that_is_not_verbatim_is_refused_naming_it():
    # this is a unit test
    """Every lexical measurement in this lab — quote recall, the Inspector's
    green spans, the offline RAGAS context metrics — is computed against these
    evidence quotes, so a dataset that misquotes its own corpus would score
    *confidently* about text that was never there."""
    corpus, ground_truth = _valid_pair()
    ground_truth['groundtruth_dataset'][0]['relevant_corpus_documents'][0][
        'evidence'][0]['text'] = 'the roof was fixed in April'
    problems = datasets.validate(corpus, ground_truth)
    assert any('verbatim' in p for p in problems), problems


def test_a_question_citing_a_missing_document_is_refused():
    # this is a unit test
    corpus, ground_truth = _valid_pair()
    ground_truth['groundtruth_dataset'][0]['relevant_corpus_documents'][0][
        'corpus_document_id'] = 99
    problems = datasets.validate(corpus, ground_truth)
    assert any('not in this corpus' in p for p in problems), problems


def test_a_mismatched_pair_join_is_refused():
    # this is a unit test
    """D1: the two files are paired by id, not by filename — so the id each
    one declares has to actually agree."""
    corpus, ground_truth = _valid_pair()
    ground_truth['groundtruth_dataset_metadata']['corpus_ref']['dataset'] = 'other'
    problems = datasets.validate(corpus, ground_truth)
    assert any('pair join' in p for p in problems), problems


def test_every_problem_is_reported_at_once():
    # this is a unit test
    """One problem per attempt is a slow loop over a 200-question corpus."""
    corpus, ground_truth = _valid_pair()
    ground_truth['groundtruth_dataset_metadata']['corpus_ref']['dataset'] = 'other'
    ground_truth['groundtruth_dataset'][0]['relevant_corpus_documents'][0][
        'corpus_document_id'] = 99
    problems = datasets.validate(corpus, ground_truth)
    assert len(problems) >= 2, problems


def test_an_extracted_label_needs_exactly_one_rater():
    # this is a unit test
    """D9/x-consistency: `extracted:true` means the values can be wrong, so a
    caveat has to exist against them."""
    corpus, ground_truth = _valid_pair(corpus_overrides={
        'corpus_dataset_metadata': {
            'dataset': 'tiny-test', 'name': 'Tiny', 'language': 'en',
            'label_fields': {
                'feeling': {'type': 'string', 'description': 'mood',
                           'applies_to': ['document'], 'extracted': True}}}})
    corpus['corpus_documents'][0]['document_metadata'] = {'feeling': 'glad'}
    problems = datasets.validate(corpus, ground_truth)
    assert any('exactly one' in p for p in problems), problems


def test_a_rater_may_not_rate_an_unextracted_label():
    # this is a unit test
    corpus, ground_truth = _valid_pair(corpus_overrides={
        'corpus_dataset_metadata': {
            'dataset': 'tiny-test', 'name': 'Tiny', 'language': 'en',
            'label_fields': {
                'topic': {'type': 'string', 'description': 'subject',
                         'applies_to': ['document']},
                'topic_confidence': {
                    'type': 'number', 'description': 'how sure',
                    'applies_to': ['document'], 'confidence_for': 'topic'}}}})
    corpus['corpus_documents'][0]['document_metadata'] = {
        'topic': 'roofing', 'topic_confidence': 0.9}
    problems = datasets.validate(corpus, ground_truth)
    assert any('rates' in p and 'not declared extracted' in p for p in problems), (
        problems)


def test_ranks_needs_a_numeric_type_with_bounds():
    # this is a unit test
    """D6: importance's one declared source. A label that ranks must be
    numeric and bounded, or "rescaled to 0-1" has nothing to rescale from."""
    corpus, ground_truth = _valid_pair(corpus_overrides={
        'corpus_dataset_metadata': {
            'dataset': 'tiny-test', 'name': 'Tiny', 'language': 'en',
            'label_fields': {
                'urgency': {'type': 'string', 'description': 'how urgent',
                           'applies_to': ['document'], 'ranks': True}}}})
    problems = datasets.validate(corpus, ground_truth)
    assert any('ranks' in p and 'number or integer' in p for p in problems), (
        problems)


def test_a_rater_may_never_also_rank():
    # this is a unit test
    corpus, ground_truth = _valid_pair(corpus_overrides={
        'corpus_dataset_metadata': {
            'dataset': 'tiny-test', 'name': 'Tiny', 'language': 'en',
            'label_fields': {
                'topic': {'type': 'string', 'description': 'subject',
                         'applies_to': ['document'], 'extracted': True},
                'topic_confidence': {
                    'type': 'number', 'description': 'how sure', 'minimum': 0,
                    'maximum': 1, 'applies_to': ['document'],
                    'confidence_for': 'topic', 'ranks': True}}}})
    problems = datasets.validate(corpus, ground_truth)
    assert any('ranks and' in p for p in problems), problems


def test_a_derived_fact_with_no_supporting_evidence_is_refused():
    # this is a unit test
    corpus, ground_truth = _valid_pair()
    question = ground_truth['groundtruth_dataset'][0]
    question['expected_answer']['derived_facts'] = [
        {'derived_fact_id': 1, 'fact': 'The roof was fixed'},
        {'derived_fact_id': 2, 'fact': 'It was fixed on 3 March'}]
    question['relevant_corpus_documents'][0]['evidence'][0]['supports'] = [1]
    problems = datasets.validate(corpus, ground_truth)
    assert any('derived_fact_id [2]' in p for p in problems), problems


def test_a_supports_id_that_names_no_derived_fact_is_refused():
    # this is a unit test
    corpus, ground_truth = _valid_pair()
    question = ground_truth['groundtruth_dataset'][0]
    question['relevant_corpus_documents'][0]['evidence'][0]['supports'] = [7]
    problems = datasets.validate(corpus, ground_truth)
    assert any('not derived_fact_ids' in p for p in problems), problems


def test_computed_evidence_needs_a_relevant_metadata_source():
    # this is a unit test
    corpus, ground_truth = _valid_pair()
    ground_truth['groundtruth_dataset'][0]['relevant_corpus_documents'][0][
        'evidence'][0]['fidelity'] = 'computed'
    problems = datasets.validate(corpus, ground_truth)
    assert any('relevant_metadata naming the label' in p for p in problems), (
        problems)


def test_relevant_metadata_naming_a_value_the_document_does_not_carry_is_refused():
    # this is a unit test
    """x-cross-file #6: a value named in relevant_metadata has to be one the
    document actually holds under that label — not merely a label the
    document happens to declare somewhere."""
    corpus, ground_truth = _valid_pair(corpus_overrides={
        'corpus_dataset_metadata': {
            'dataset': 'tiny-test', 'name': 'Tiny', 'language': 'en',
            'label_fields': {
                'topic': {'type': 'string', 'description': 'subject',
                         'applies_to': ['document']}}}})
    corpus['corpus_documents'][0]['document_metadata'] = {'topic': 'roofing'}
    ground_truth['groundtruth_dataset'][0]['relevant_corpus_documents'][0][
        'evidence'][0]['relevant_metadata'] = {'topic': 'plumbing'}
    problems = datasets.validate(corpus, ground_truth)
    assert any('question 1' in p and "'topic'='plumbing'" in p
              and 'does not match' in p for p in problems), problems


def test_a_copied_document_metadata_that_disagrees_with_the_corpus_is_refused():
    # this is a unit test
    """x-cross-file: 'where a piece of evidence copies the document_metadata,
    the copy must equal what the corpus holds for that document.'"""
    corpus, ground_truth = _valid_pair()
    corpus['corpus_documents'][0]['document_metadata'] = {'topic': 'roofing'}
    ground_truth['groundtruth_dataset'][0]['relevant_corpus_documents'][0][
        'evidence'][0]['document_metadata'] = {'topic': 'plumbing'}
    problems = datasets.validate(corpus, ground_truth)
    assert any('question 1' in p and 'document_metadata does not match'
              in p for p in problems), problems


def test_verbatim_evidence_with_no_part_labels_is_refused():
    # this is a unit test
    corpus, ground_truth = _valid_pair()
    ground_truth['groundtruth_dataset'][0]['relevant_corpus_documents'][0][
        'evidence'][0]['part_labels'] = []
    problems = datasets.validate(corpus, ground_truth)
    assert any('question 1' in p and 'needs part_labels' in p
              for p in problems), problems


def test_computed_evidence_with_part_labels_is_refused():
    # this is a unit test
    corpus, ground_truth = _valid_pair()
    evidence = ground_truth['groundtruth_dataset'][0][
        'relevant_corpus_documents'][0]['evidence'][0]
    evidence['fidelity'] = 'computed'
    evidence['relevant_metadata'] = {'topic': 'roofing'}
    evidence['part_labels'] = [{}]
    problems = datasets.validate(corpus, ground_truth)
    assert any('question 1' in p and 'part_labels must be empty' in p
              for p in problems), problems


def test_a_derived_facts_relevant_metadata_naming_an_uncarried_value_is_refused():
    # this is a unit test
    """The x-cross-file #6 placement D9 also names: a derived fact's own
    `relevant_metadata` — not tied to one document the way a piece of
    evidence is — still has to name a value some document the question cites
    actually carries."""
    corpus, ground_truth = _valid_pair(corpus_overrides={
        'corpus_dataset_metadata': {
            'dataset': 'tiny-test', 'name': 'Tiny', 'language': 'en',
            'label_fields': {
                'topic': {'type': 'string', 'description': 'subject',
                         'applies_to': ['document']}}}})
    corpus['corpus_documents'][0]['document_metadata'] = {'topic': 'roofing'}
    question = ground_truth['groundtruth_dataset'][0]
    question['expected_answer']['derived_facts'] = [
        {'derived_fact_id': 1, 'fact': 'The roof was fixed',
         'relevant_metadata': {'topic': 'plumbing'}}]
    problems = datasets.validate(corpus, ground_truth)
    assert any('question 1' in p and 'derived_fact 1' in p
              and "'topic'='plumbing'" in p and 'not carried by any document' in p
              for p in problems), problems


def test_a_label_shared_by_both_files_with_a_different_type_is_refused_naming_it():
    # this is a unit test
    """x-cross-file #10: 'A label declared in both files must carry the same
    meaning in both.' A label the corpus calls a string and the ground truth
    calls a number is not one shared vocabulary entry — it is two different
    fields that happen to share a spelling."""
    corpus, ground_truth = _valid_pair(corpus_overrides={
        'corpus_dataset_metadata': {
            'dataset': 'tiny-test', 'name': 'Tiny', 'language': 'en',
            'label_fields': {
                'severity': {'type': 'string', 'description': 'how bad',
                            'applies_to': ['document']}}}})
    ground_truth['groundtruth_dataset_metadata']['question_metadata_fields'] = {
        'severity': {'type': 'number', 'description': 'how bad',
                     'applies_to': ['question']}}
    problems = datasets.validate(corpus, ground_truth)
    assert any("label 'severity'" in p and 'does not carry the same meaning' in p
              and 'type' in p for p in problems), problems


def test_a_label_shared_by_both_files_with_a_different_closed_set_is_refused_naming_it():
    # this is a unit test
    """The same rule on the other honest slice of 'meaning': a label closed
    to a different set of values in each file is not agreeing on what it
    means, even though both sides call it a string."""
    corpus, ground_truth = _valid_pair(corpus_overrides={
        'corpus_dataset_metadata': {
            'dataset': 'tiny-test', 'name': 'Tiny', 'language': 'en',
            'label_fields': {
                'severity': {'type': 'string', 'description': 'how bad',
                            'applies_to': ['document'],
                            'values': ['low', 'high']}}}})
    ground_truth['groundtruth_dataset_metadata']['question_metadata_fields'] = {
        'severity': {'type': 'string', 'description': 'how bad',
                     'applies_to': ['question'], 'values': ['low', 'medium', 'high']}}
    problems = datasets.validate(corpus, ground_truth)
    assert any("label 'severity'" in p and 'does not carry the same meaning' in p
              and 'allowed values' in p for p in problems), problems


def test_a_label_shared_by_both_files_agreeing_on_meaning_is_clean():
    # this is a unit test
    corpus, ground_truth = _valid_pair(corpus_overrides={
        'corpus_dataset_metadata': {
            'dataset': 'tiny-test', 'name': 'Tiny', 'language': 'en',
            'label_fields': {
                'severity': {'type': 'string', 'description': 'how bad',
                            'applies_to': ['document'],
                            'values': ['low', 'high']}}}})
    ground_truth['groundtruth_dataset_metadata']['question_metadata_fields'] = {
        'severity': {'type': 'string', 'description': 'how bad',
                     'applies_to': ['question'], 'values': ['low', 'high']}}
    assert datasets.validate(corpus, ground_truth) == []


def test_an_abstention_question_needs_no_evidence():
    # this is a unit test
    """Abstention questions are the ones the corpus deliberately cannot
    answer: demanding evidence for them would make the failure mode the
    relevance gate exists for unmeasurable."""
    corpus, ground_truth = _valid_pair()
    ground_truth['groundtruth_dataset'].append({
        'groundtruth_question_id': 2, 'question': 'Who paid for the roof?',
        'expected_answer': {'behavior': 'abstain'},
        'relevant_corpus_documents': []})
    assert datasets.validate(corpus, ground_truth) == []


def test_the_schemas_own_x_authoring_examples_validate():
    # this is a unit test
    """`schema_corpus.json`'s `smallest_valid_corpus` and
    `schema_groundtruth.json`'s `smallest_valid_question_set` are the first
    thing an author copies (`x-authoring.steps`: "Start with the smallest
    thing that validates, run it..."). An example that does not itself pass
    `validate()` teaches the wrong lesson before a real dataset is written,
    so the schemas' own worked example is pinned here the same way the
    bundled pairs are below."""
    corpus = datasets.CORPUS_SCHEMA['x-authoring']['smallest_valid_corpus']
    ground_truth = datasets.GROUNDTRUTH_SCHEMA['x-authoring'][
        'smallest_valid_question_set']
    assert datasets.validate(corpus, ground_truth) == []


# --- the bundled samples -----------------------------------------------------

def test_the_bundled_datasets_meet_their_contract_and_cover_the_failure_modes():
    # this is a unit test
    """These are reference points — the corpora a finding is checked against
    to tell "true of retrieval" from "true of Farsi diaries" (or of German
    meetings, or of multi-hop research notes). Each must actually validate,
    and between them they have to offer more than one language and questions
    the corpus cannot answer — otherwise they are several spellings of the
    same test."""
    languages = set()
    for dataset_id in CONTROLS:
        corpus, ground_truth = datasets.load(dataset_id)
        problems = datasets.validate(corpus, ground_truth)
        assert problems == [], (dataset_id, problems)
        languages.add(corpus['corpus_dataset_metadata']['language'])
        behaviors = {q['expected_answer']['behavior']
                    for q in ground_truth['groundtruth_dataset']}
        assert {'answer', 'abstain', 'correct_premise'} <= behaviors, dataset_id
    assert len(languages) >= 2, languages


def test_the_catalogue_lists_every_bundled_dataset():
    # this is a unit test
    found = datasets.catalogue()
    ids = {d.id for d in found}
    assert set(BUNDLED) <= ids
    for entry in found:
        if entry.id in BUNDLED:
            assert entry.source == 'bundled', entry.id


# --- loading ------------------------------------------------------------------

def test_a_loaded_dataset_arrives_in_the_schema_shape_and_nothing_else():
    # this is a unit test
    """D4: `load()` returns the two file payloads exactly as they are — there
    is no second, translated dialect."""
    corpus, ground_truth = datasets.load('smoke-mini')
    assert corpus['corpus_documents'] and ground_truth['groundtruth_dataset']
    question = ground_truth['groundtruth_dataset'][0]
    assert 'question' in question and 'question_fa' not in question
    assert question['expected_answer']['behavior'] in (
        'answer', 'abstain', 'correct_premise')
    document = corpus['corpus_documents'][0]
    assert 'document_content' in document
    assert 'sessions' not in corpus and 'messages' not in document


def test_an_unknown_dataset_says_what_there_is():
    # this is a unit test
    with pytest.raises(ValueError) as raised:
        datasets.load('not-a-corpus')
    assert 'smoke-mini' in str(raised.value)


# --- D1: a dataset is two files, paired by id --------------------------------

def test_a_corpus_with_no_ground_truth_is_listed_and_refused_at_load_time(
        imports_here):
    # this is an integration test
    imports_here.mkdir(parents=True)
    corpus, _ = _valid_pair()
    (imports_here / 'tiny-test_corpus.json').write_text(
        json.dumps(corpus), encoding='utf-8')

    listed = next(d for d in datasets.catalogue() if d.id == 'tiny-test')
    assert listed.questions == 0
    with pytest.raises(ValueError, match='nothing to measure against'):
        datasets.load('tiny-test')


def test_a_corpus_with_no_ground_truth_can_still_be_described(imports_here):
    # this is an integration test
    """The other half of the rule above: refused at run time, but still
    *readable*. The panel's dataset card reads a listed corpus's own
    `label_fields`, so one unmeasurable corpus in the folder must not take the
    whole catalogue down with it — `load_corpus` reads what `load` refuses,
    and nothing that scores goes through it."""
    imports_here.mkdir(parents=True)
    corpus, _ = _valid_pair()
    (imports_here / 'tiny-test_corpus.json').write_text(
        json.dumps(corpus), encoding='utf-8')

    assert datasets.load_corpus('tiny-test') == corpus
    with pytest.raises(ValueError, match='unknown dataset'):
        datasets.load_corpus('not-a-corpus')


def test_a_ground_truth_with_no_corpus_is_never_listed(imports_here):
    # this is an integration test
    imports_here.mkdir(parents=True)
    _, ground_truth = _valid_pair()
    (imports_here / 'tiny-test_groundtruth.json').write_text(
        json.dumps(ground_truth), encoding='utf-8')

    ids = {d.id for d in datasets.catalogue()}
    assert 'tiny-test' not in ids


# --- the index cannot mix corpora --------------------------------------------

def test_the_dataset_is_part_of_the_fingerprint_and_the_blank_case_is_pinned():
    # this is a unit test
    """The one bug this feature could plausibly introduce, and the most
    expensive to notice late: an index built over one corpus handed a question
    from another. `''` is fingerprinted exactly as it was before the field
    existed (D3) — every run in `.runs/` records a collection name, and a new
    field in IndexConfig would otherwise rename them all."""
    assert (IndexConfig(dataset='smoke-mini').fingerprint()
            != IndexConfig().fingerprint())
    assert (IndexConfig(dataset='smoke-mini').fingerprint()
            != IndexConfig(dataset='support-en').fingerprint())
    # The literal is the value the field-less IndexConfig produced, read off
    # the code as it stood before this commit rather than off the code it
    # is checking.
    assert IndexConfig().fingerprint() == '6cf7db2bab4f'
    assert IndexConfig(dataset='').collection() == 'raglab-6cf7db2bab4f'


# --- a dataset entering the lab is stored as content -------------------------

@pytest.fixture
def corpora_here(tmp_path, monkeypatch):
    """This test's own corpus store. The suite's autouse redirect already keeps
    every test off the developer's file; this narrows it to one file per test,
    so counting rows means counting *this* test's rows."""
    monkeypatch.setenv('RAGLAB_CORPORA_DB', str(tmp_path / 'corpora.db'))
    return tmp_path / 'corpora.db'


def _stored_rows(path) -> list[dict]:
    with corpora.connect(path) as db:
        return [dict(row) for row in db.execute(
            'SELECT id, dataset, corpus, ground_truth FROM corpora ORDER BY id')]


def test_an_imported_dataset_is_stored_as_content_under_an_id_the_file_never_carried(
        imports_here, corpora_here):
    # this is an integration test
    """A dataset entering the lab enters the corpus store, once, on arrival.

    The two objects stored are exactly the two file payloads the pipeline
    reads, so the row written on import is the row every later experiment on
    this corpus references, rather than one written again per archive. The id
    is the database's to give: the file carries none and could not, because an
    id is storage identity on this machine and a file that named one would be
    claiming a row it knows nothing about.
    """
    corpus, ground_truth = _valid_pair()
    assert 'id_corpora' not in json.dumps(corpus)
    found = datasets.import_dataset(corpus, ground_truth)

    assert isinstance(found.id_corpora, int) and found.id_corpora > 0
    assert found.as_dict()['id_corpora'] == found.id_corpora
    rows = _stored_rows(corpora_here)
    assert [row['id'] for row in rows] == [found.id_corpora]
    assert rows[0]['dataset'] == 'tiny-test'
    # What the store holds is what the lab loads: the same two objects, not a
    # second reading of the file that could differ from the one that runs.
    assert corpora.get(found.id_corpora) == datasets.load('tiny-test')
    # And the files still work exactly as they did.
    assert (imports_here / 'tiny-test_corpus.json').exists()
    assert (imports_here / 'tiny-test_groundtruth.json').exists()


def test_importing_the_same_dataset_twice_stores_one_corpus(imports_here,
                                                            corpora_here):
    # this is an integration test
    """Idempotent, because the content decides. Re-importing a pair that has
    not changed is not a new corpus, and a second row would make it look like
    one to everything that later joins on the id."""
    first = datasets.import_dataset(*_valid_pair())
    second = datasets.import_dataset(*_valid_pair())
    assert first.id_corpora == second.id_corpora
    assert len(_stored_rows(corpora_here)) == 1


def test_an_edited_dataset_under_the_same_id_is_a_new_row_beside_the_old_one(
        imports_here, corpora_here):
    # this is an integration test
    """The claim an older archive depends on.

    A corpus edited between runs is a *different* corpus. The files at
    `.datasets/tiny-test_*.json` are replaced — that is what the reader asked
    for — but the stored text is not: the edit is a new row with a new id, and
    the row the earlier experiment referenced is left exactly as it was.
    """
    before = datasets.import_dataset(*_valid_pair())
    edited_corpus, edited_truth = _valid_pair()
    edited_corpus['corpus_documents'][0]['document_content'][0]['text'] = (
        'Actually the roof was fixed on 3 March, twice over.')
    after = datasets.import_dataset(edited_corpus, edited_truth)

    assert after.id_corpora != before.id_corpora
    rows = _stored_rows(corpora_here)
    assert [row['id'] for row in rows] == [before.id_corpora, after.id_corpora]
    old_corpus, _ = corpora.get(before.id_corpora)
    new_corpus, _ = corpora.get(after.id_corpora)
    assert old_corpus['corpus_documents'][0]['document_content'][0][
        'text'].startswith('Good news')
    assert new_corpus['corpus_documents'][0]['document_content'][0][
        'text'].startswith('Actually')
    assert {version['id'] for version in corpora.versions('tiny-test')} == \
        {before.id_corpora, after.id_corpora}


def test_a_listing_says_it_did_not_look_the_corpus_row_up(imports_here,
                                                          corpora_here):
    # this is an integration test
    """`id_corpora` on a catalogue row is 0, and that means "not asked".

    A listing reads files; it resolves nothing against the corpus store. The
    honest value for a lookup that never happened is the one no corpora row can
    ever have — `AUTOINCREMENT` starts at 1 — so a reader cannot mistake it for
    a row that says there is no stored corpus.
    """
    stored = datasets.import_dataset(*_valid_pair())
    assert stored.id_corpora > 0
    listed = next(d for d in datasets.catalogue() if d.id == 'tiny-test')
    assert listed.id_corpora == 0
    assert datasets.find('tiny-test').id_corpora == 0


def test_an_import_that_breaks_the_contract_is_refused_with_every_reason(
        imports_here):
    # this is an integration test
    corpus, ground_truth = _valid_pair()
    ground_truth['groundtruth_dataset'][0]['relevant_corpus_documents'][0][
        'evidence'][0]['text'] = 'never said this'
    ground_truth['groundtruth_dataset_metadata']['corpus_ref']['dataset'] = 'other'
    with pytest.raises(ValueError) as raised:
        datasets.import_dataset(corpus, ground_truth)
    detail = str(raised.value)
    assert 'verbatim' in detail and 'pair join' in detail


def test_the_built_in_corpus_cannot_be_overwritten_by_an_import(imports_here):
    # this is an integration test
    corpus, ground_truth = _valid_pair()
    corpus['corpus_dataset_metadata']['dataset'] = datasets.BUILTIN
    ground_truth['groundtruth_dataset_metadata']['corpus_ref']['dataset'] = (
        datasets.BUILTIN)
    with pytest.raises(ValueError, match='built-in'):
        datasets.import_dataset(corpus, ground_truth)


# --- the leaderboard ----------------------------------------------------------

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
    assert 'support-en' in leaderboard.markdown(leaderboard.by_dataset(rows))


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


# --- the panel's own conventions, unrelated to loading -----------------------

def test_the_panel_renders_the_json_shape_as_a_shape():
    # this is a convention test
    """This is the one help text with a structure in it, and a structure that
    arrives as one run-on paragraph is not one. Every other explainer is a
    single line, so preserving the newlines costs them nothing."""
    from raglab.dashboard.service_route_plumbing import STATIC
    css = (STATIC / 'panel.css').read_text(encoding='utf-8')
    rule = css.split('p.explain {')[1].split('}')[0]
    assert 'pre-wrap' in rule


def test_the_panel_offers_the_dataset_and_ranks_per_corpus():
    # this is a convention test
    """Two corpora are never one measurement, so a ranking must never span
    them. This checks the dataset is still offered here and that the rule is
    still enforced in the grouping module, rather than checking for markup
    that moved."""
    from raglab.dashboard.service_route_plumbing import STATIC
    html = (STATIC / 'panel.html').read_text(encoding='utf-8')
    js = (STATIC / 'panel.js').read_text(encoding='utf-8')
    # A dataset is two files, paired by id (D1): a corpus and its ground
    # truth, so the import control is two file inputs, not one.
    assert ('id="dataset"' in html and 'id="dataset-corpus-file"' in html
            and 'id="dataset-groundtruth-file"' in html)
    assert '/api/datasets' in js
    rows = [{'dataset': 'diary-fa', 'label': 'a',
             'selection': {'question_ids': ['q1']}, 'judge': {'model': 'm'}},
            {'dataset': 'smoke-mini', 'label': 'b',
             'selection': {'question_ids': ['q1']}, 'judge': {'model': 'm'}}]
    assert len(leaderboard.group(rows)) == 2, (
        'two corpora must never share a table, even on identical questions '
        'with the same judge')
