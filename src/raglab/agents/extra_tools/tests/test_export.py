"""Exporting a run for reading — one Markdown page per question plus an
index, built only from what a finished run stored."""
import json

from raglab.agents.extra_tools import export


# --- exporting a run for reading ------------------------------------------
# The leaderboard cannot show what the pipeline did to any one question; the
# export writes that out from a finished run, only from what it stored.
#
# A run file's rows carry no question label of their own (D7's rows are
# scored, not described) — `question_type`/`difficulty` are joined in from
# the ground truth's own `question_metadata`, keyed by the row's `id`, which
# is the `groundtruth_question_id` an evaluation actually stores.

GROUND_TRUTH = {
    'groundtruth_dataset_metadata': {
        'name': 'export fixture', 'corpus_ref': {'dataset': 'export-test'}},
    'groundtruth_dataset': [
        {'groundtruth_question_id': 1, 'question': 'چند بار در هفته؟',
         'expected_answer': {'behavior': 'answer', 'text': 'هفته‌ای سه بار.'},
         'relevant_corpus_documents': [
             {'corpus_document_id': 42,
              'evidence': [{'text': 'سه بار در هفته می‌رود', 'fidelity': 'verbatim'}]}],
         'question_metadata': {'question_type': 'habit', 'difficulty': 'medium'}},
        {'groundtruth_question_id': 2, 'question': 'چیزی که هرگز نگفته؟',
         'expected_answer': {'behavior': 'abstain'},
         'relevant_corpus_documents': [],
         'question_metadata': {'question_type': 'abstention', 'difficulty': 'hard'}},
        {'groundtruth_question_id': 3, 'question': 'تیر چه اتفاقی افتاد؟',
         'expected_answer': {'behavior': 'answer', 'text': 'نمی‌دانم.'},
         'relevant_corpus_documents': [
             {'corpus_document_id': 7,
              'evidence': [{'text': 'یک اتفاق در تیر', 'fidelity': 'verbatim'}]}],
         'question_metadata': {'question_type': 'single-hop', 'difficulty': 'easy',
                               'resolved_time_scope': 'تیر'}},
    ],
}
QUESTIONS = {q['groundtruth_question_id']: q
            for q in GROUND_TRUTH['groundtruth_dataset']}

RUN_FIXTURE = {
    'run_id': '20260101-010101-abc123', 'label': 'D wider context k=12',
    'seconds': 671.68, 'started_at': '2026-01-01 01:01:01',
    'config': {'index': {'split_plan': [{'kind': 'document'}, {'kind': 'drift', 'markers': [], 'when': 'always'}], 'embedder': 'x',
                         'embed_model': 'y'},
               'retrieval': {'k': 12, 'retriever': 'hybrid-rrf',
                             'reranker': 'lexical'},
               'generation': {'answerer': 'llm', 'model': 'm'}},
    'summary': {'n_questions': 3, 'overall': {}},
    'ragas': {'mode': 'llm', 'metrics': {'faithfulness': 0.77},
              'decision': 0.6501, 'n_samples': 3,
              'decision_spread': {'n': 3, 'mean': 0.65, 'stderr': 0.04}},
    'rows': [
        {'id': 1, 'behavior': 'answer', 'abstained': False, 'hit': 1.0,
         'recall': 1.0, 'quote_recall': 1.0, 'ndcg': 0.63, 'precision': 0.2,
         'mrr': 0.5, 'n_contexts': 8, 'context_chars': 4689,
         'latency_ms': 21140.4, 'time_scope': None,
         'retrieved_sessions': ['42', '99'],
         'answer': 'هفته‌ای سه بار.', 'answer_similarity': 0.31,
         'answer_token_f1': 0.36},
        {'id': 2, 'behavior': 'abstain', 'abstained': True, 'hit': None,
         'recall': None, 'quote_recall': None, 'n_contexts': 8,
         'context_chars': 100, 'latency_ms': 900.0, 'time_scope': None,
         'retrieved_sessions': [], 'answer': 'پیدا نکردم.'},
        {'id': 3, 'behavior': 'answer', 'abstained': False, 'hit': 0.0,
         'recall': 0.0, 'quote_recall': 0.0, 'n_contexts': 8,
         'context_chars': 300, 'latency_ms': 800.0, 'time_scope': 'تیر',
         'retrieved_sessions': ['7'], 'answer': 'نمی‌دانم.'},
    ],
}


def test_the_difficulty_table_counts_answers_and_evidence_separately():
    # this is a unit test
    """The run files store no judged grade per question, so "correct" is
    evidence-based: an answerable question counts when the pipeline did not
    refuse *and* reached a gold document; an unanswerable one counts when it
    did refuse. Evidence reaching context and the answer using it are
    different failures, so they are reported apart rather than collapsed —
    and an unanswerable question has no evidence to find, so it must not be
    averaged in as a miss."""
    table = export.difficulty_rates(RUN_FIXTURE['rows'], QUESTIONS)
    assert [row['difficulty'] for row in table] == ['medium', 'hard', 'easy']
    medium, hard, easy = table
    assert easy['n'] == 1 and easy['answered'] == 0.0     # retrieved nothing gold
    assert medium['n'] == 1 and medium['answered'] == 1.0
    assert hard['n'] == 1 and hard['answered'] == 1.0     # correctly refused
    # The share is a share of that difficulty, so it needs the count beside it:
    # one hard question at 100% is not a finding.
    assert all('n' in row for row in table)
    assert easy['evidence_found'] == 0.0
    assert easy['quotes_in_context'] == 0.0
    assert hard['evidence_found'] is None


def test_a_question_page_shows_everything_needed_to_judge_one_question():
    # this is a unit test
    """The four things you need to judge one question, in one file — plus the
    two disclaimers a bare number would hide: the four deciding metrics are
    run means only, and the retrieved context is document ids, never chunk
    text the run never stored (runs store the retrieved document ids, not the
    chunk text, so the page says what it has rather than reconstructing
    chunks by re-running retrieval, which would document a different
    retrieval than the one that was graded)."""
    question = QUESTIONS[3]
    row = next(r for r in RUN_FIXTURE['rows'] if r['id'] == 3)
    page = export.question_page(RUN_FIXTURE, question, row)
    assert question['question'] in page
    assert question['expected_answer']['text'] in page      # the reference
    evidence = question['relevant_corpus_documents'][0]['evidence'][0]
    assert evidence['text'] in page                          # and its quote
    assert str(question['relevant_corpus_documents'][0]['corpus_document_id']) in page
    assert row['answer'] in page                             # what it replied
    assert '7' in page                                        # what it retrieved
    # Every grade names its own arithmetic, the same rule the dashboard follows.
    assert '|gold ∩ top-k| / |gold|' in page
    assert 'Recall@k' in page
    # And the run it came from, or the page cannot be traced back.
    assert RUN_FIXTURE['run_id'] in page and RUN_FIXTURE['label'] in page
    # The four deciding metrics are stored as run means only; printed
    # unlabelled they would read as that question's own faithfulness.
    assert 'run mean' in page.lower()
    assert 'not per question' in page.lower()
    assert 'chunk text is not stored' in page.lower()


def test_the_export_writes_one_file_per_question_plus_an_index(tmp_path):
    # this is an integration test
    written = export.write_run(RUN_FIXTURE, GROUND_TRUTH, tmp_path)
    names = sorted(path.name for path in written)
    assert names == ['1.md', '2.md', '3.md', 'README.md']
    index = (tmp_path / 'README.md').read_text(encoding='utf-8')
    assert 'easy' in index and 'medium' in index and 'hard' in index
    assert RUN_FIXTURE['run_id'] in index
    # The index links the files, or a folder of 24 pages is unnavigable.
    assert '(3.md)' in index
    # And a written page is really the page, not an empty file `write_run`
    # happened to create at the right name.
    page = (tmp_path / '3.md').read_text(encoding='utf-8')
    assert 'chunk text is not stored' in page.lower()


# --- the command line -----------------------------------------------------
# `raglab-export` only resolves an input to a (run, ground truth) pair and
# hands both to `write_run` — the same renderer the tests above pin. Two
# input kinds: a run file from `.runs/`, whose ground truth is the dataset it
# names, and an exported experiment archive, which carries its own. Neither
# path may re-run retrieval or derive a number, so the risk worth testing is
# the resolution and the refusal, not the pages.

def _cli(capsys, *argv) -> tuple[int, str, str]:
    """Run the command in-process and return `(exit code, stdout, stderr)`."""
    try:
        export.main(list(argv))
        code = 0
    except SystemExit as stop:
        code = 1 if stop.code is None else stop.code
    out, err = capsys.readouterr()
    return code, out, err


def test_the_command_explains_its_inputs_outputs_and_an_invocation(capsys):
    # this is a unit test
    code, out, err = _cli(capsys, '--help')
    assert code == 0
    assert '--out-dir' in out
    assert '.runs/' in out and 'archive' in out
    assert 'raglab-export' in out            # a copy-pasteable invocation


def test_the_command_exports_a_run_file_against_the_dataset_it_names(
        tmp_path, capsys):
    # this is an integration test
    """A run file stores no ground truth, only the dataset id — so the pages
    are joined against the corpus the run itself names, through the same
    loader an evaluation used."""
    run = dict(RUN_FIXTURE, dataset='smoke-mini')
    source = tmp_path / f"{run['run_id']}.json"
    source.write_text(json.dumps(run), encoding='utf-8')
    out_dir = tmp_path / 'pages'
    code, out, err = _cli(capsys, str(source), '--out-dir', str(out_dir))
    assert code == 0
    # Machine-readable: the folder it wrote, one line, nothing else on stdout.
    assert out.strip() == str(out_dir)
    assert sorted(path.name for path in out_dir.glob('*.md')) == [
        '1.md', '2.md', '3.md', 'README.md']
    assert run['run_id'] in (out_dir / 'README.md').read_text(encoding='utf-8')


def test_the_command_exports_an_archive_from_the_ground_truth_it_carries(
        tmp_path, capsys):
    # this is an integration test
    """An exported archive is self-contained, and it is read through the very
    codec the panel's import uses — so a file this installation's dataset
    folder knows nothing about still exports."""
    from raglab.evaluation.tests import archive_examples

    source = tmp_path / 'archive.json'
    source.write_text(json.dumps(archive_examples.completed_archive()),
                      encoding='utf-8')
    out_dir = tmp_path / 'pages'
    code, out, err = _cli(capsys, str(source), '--out-dir', str(out_dir))
    assert code == 0
    assert sorted(path.name for path in out_dir.glob('*.md')) == [
        '1.md', 'README.md']
    assert 'imported-run-001' in (out_dir / '1.md').read_text(encoding='utf-8')


def test_the_command_refuses_rather_than_writing_half_a_report(tmp_path, capsys):
    # this is an integration test
    """Every refusal is non-zero, says why on stderr, keeps stdout empty and
    leaves the output folder uncreated — a half-written report would be read
    as a whole one."""
    out_dir = tmp_path / 'pages'

    # No input at all: argparse's own usage error, before anything is read.
    code, out, err = _cli(capsys, '--out-dir', str(out_dir))
    assert code == 2 and 'usage:' in err and out == ''

    # A file that is neither a run nor an archive.
    stranger = tmp_path / 'notes.json'
    stranger.write_text('{"hello": "world"}', encoding='utf-8')
    code, out, err = _cli(capsys, str(stranger), '--out-dir', str(out_dir))
    assert code != 0 and out == ''
    assert 'run file' in err and 'archive' in err

    # A run whose rows the dataset's questions do not include: there is
    # nothing to report, and reporting an empty index would claim otherwise.
    empty = tmp_path / 'empty.json'
    empty.write_text(json.dumps(
        dict(RUN_FIXTURE, dataset='smoke-mini', rows=[])), encoding='utf-8')
    code, out, err = _cli(capsys, str(empty), '--out-dir', str(out_dir))
    assert code != 0 and out == ''
    assert 'no question' in err

    assert not out_dir.exists()
