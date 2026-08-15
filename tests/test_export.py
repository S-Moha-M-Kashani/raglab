"""Exporting a run for reading — one Markdown page per question plus an
index, built only from what a finished run stored."""
from raglab.llm_tools import export


# --- exporting a run for reading ------------------------------------------
# The leaderboard cannot show what the pipeline did to any one question; the
# export writes that out from a finished run, only from what it stored.

RUN_FIXTURE = {
    'run_id': '20260101-010101-abc123', 'label': 'D wider context k=12',
    'seconds': 671.68, 'started_at': '2026-01-01 01:01:01',
    'config': {'index': {'chunker': 'semantic-drift', 'embedder': 'x',
                         'embed_model': 'y', 'layers': ['chunk', 'habit']},
               'retrieval': {'k': 12, 'retriever': 'hybrid-rrf',
                             'reranker': 'lexical'},
               'generation': {'answerer': 'llm', 'model': 'm'}},
    'index': {'chunks': 732, 'by_layer': {'chunk': 700, 'habit': 5},
              'embed_dim': 1024},
    'summary': {'n_questions': 3, 'overall': {}, 'by_type': {}},
    'ragas': {'mode': 'llm', 'metrics': {'faithfulness': 0.77},
              'decision': 0.6501, 'n_samples': 3,
              'decision_spread': {'n': 3, 'mean': 0.65, 'stderr': 0.04}},
    'rows': [
        {'id': 'q-hb-001', 'type': 'habit', 'difficulty': 'medium',
         'answerable': True, 'abstained': False, 'hit': 1.0, 'recall': 1.0,
         'quote_recall': 1.0, 'ndcg': 0.63, 'precision': 0.2, 'mrr': 0.5,
         'n_contexts': 8, 'context_chars': 4689, 'latency_ms': 21140.4,
         'layers': ['chunk', 'habit'], 'time_scope': None,
         'retrieved_sessions': ['2026-05-16-a', '2025-11-08-a'],
         'answer': 'هفته‌ای سه بار.', 'answer_similarity': 0.31,
         'answer_token_f1': 0.36},
        {'id': 'q-ab-001', 'type': 'abstention', 'difficulty': 'hard',
         'answerable': False, 'abstained': True, 'hit': None, 'recall': None,
         'quote_recall': None, 'n_contexts': 8, 'context_chars': 100,
         'latency_ms': 900.0, 'layers': ['chunk'], 'time_scope': None,
         'retrieved_sessions': [], 'answer': 'پیدا نکردم.'},
        {'id': 'q-sh-001', 'type': 'single-hop', 'difficulty': 'easy',
         'answerable': True, 'abstained': False, 'hit': 0.0, 'recall': 0.0,
         'quote_recall': 0.0, 'n_contexts': 8, 'context_chars': 300,
         'latency_ms': 800.0, 'layers': ['chunk'], 'time_scope': 'تیر',
         'retrieved_sessions': ['2025-01-01-a'], 'answer': 'نمی‌دانم.'},
    ],
}


def test_the_difficulty_table_counts_answers_and_evidence_separately():
    """The run files store no judged grade per question, so "correct" is
    evidence-based: an answerable question counts when the pipeline did not
    refuse *and* reached a gold session; an unanswerable one counts when it
    did refuse. Evidence reaching context and the answer using it are
    different failures, so they are reported apart rather than collapsed —
    and an unanswerable question has no evidence to find, so it must not be
    averaged in as a miss."""
    # this is a unit test
    table = export.difficulty_rates(RUN_FIXTURE['rows'])
    assert [row['difficulty'] for row in table] == ['easy', 'medium', 'hard']
    easy, medium, hard = table
    assert easy['n'] == 1 and easy['answered'] == 0.0     # retrieved nothing gold
    assert medium['n'] == 1 and medium['answered'] == 1.0
    assert hard['n'] == 1 and hard['answered'] == 1.0     # correctly refused
    # The share is a share of that difficulty, so it needs the count beside it:
    # one hard question at 100% is not a finding.
    assert all('n' in row for row in table)
    assert easy['evidence_found'] == 0.0
    assert easy['quotes_in_context'] == 0.0
    assert hard['evidence_found'] is None


def test_a_question_page_shows_everything_needed_to_judge_one_question(
        ground_truth):
    """The four things you need to judge one question, in one file — plus the
    two disclaimers a bare number would hide: the four deciding metrics are
    run means only, and the retrieved context is session ids, never chunk
    text the run never stored (runs store the retrieved session ids, not the
    chunk text, so the page says what it has rather than reconstructing
    chunks by re-running retrieval, which would document a different
    retrieval than the one that was graded)."""
    # this is a unit test
    question = next(q for q in ground_truth['questions'] if q['id'] == 'q-sh-001')
    row = next(r for r in RUN_FIXTURE['rows'] if r['id'] == 'q-sh-001')
    page = export.question_page(RUN_FIXTURE, question, row)
    assert question['question_fa'] in page
    assert question['question_en'] in page
    assert question['answer_fa'] in page                  # the reference
    assert question['evidence'][0]['quote'] in page       # and its quote
    assert question['evidence'][0]['session_id'] in page
    assert row['answer'] in page                          # what it replied
    assert '2025-01-01-a' in page                         # what it retrieved
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


def test_the_export_writes_one_file_per_question_plus_an_index(ground_truth,
                                                              tmp_path):
    # this is an integration test
    written = export.write_run(RUN_FIXTURE, ground_truth, tmp_path)
    names = sorted(path.name for path in written)
    assert names == ['README.md', 'q-ab-001.md', 'q-hb-001.md', 'q-sh-001.md']
    index = (tmp_path / 'README.md').read_text(encoding='utf-8')
    assert 'easy' in index and 'medium' in index and 'hard' in index
    assert RUN_FIXTURE['run_id'] in index
    # The index links the files, or a folder of 24 pages is unnavigable.
    assert '(q-sh-001.md)' in index
