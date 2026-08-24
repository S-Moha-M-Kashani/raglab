"""Write a finished run out as one readable page per question — what the pipeline retrieved, said, and was graded on, argued in a way the leaderboard's single number cannot.
This module only reports what the run stored; it never re-runs retrieval or re-derives a judged score, so every number on a page is checkable against the run file it came from.
"""
from pathlib import Path

from raglab.evaluation import deterministic_metrics as metrics

# In the order a reader wants them. `metrics.MEASURES` supplies each one's
# definition, so this list holds no wording of its own.
GRADE_KEYS = ('recall', 'quote_recall', 'ndcg', 'mrr', 'precision', 'hit',
              'false_abstention', 'answer_similarity', 'answer_token_f1',
              'fact_coverage', 'latency_ms')


def _num(value, places: int = 4) -> str:
    if value is None:
        return '—'
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, (int, float)):
        return f'{float(value):.{places}f}'.rstrip('0').rstrip('.')
    return str(value)


def _numeric_mean(values: list) -> float | None:
    # Unlike metrics._mean, this keeps NaN — isinstance(v, (int, float)) does not filter it out.
    kept = [float(v) for v in values if isinstance(v, (int, float))]
    return round(sum(kept) / len(kept), 4) if kept else None


def answered_correctly(row: dict) -> bool:
    """Evidence-based correctness: answerable means not refused and retrieval
    reached a gold document; unanswerable (behavior == 'abstain') means
    refused. Claims less than "correct" since no judged per-question grade
    exists in a run file."""
    if row.get('behavior') == 'abstain':
        return bool(row.get('abstained'))
    return not row.get('abstained') and bool(row.get('hit'))


def _label(questions: dict, row: dict, name: str):
    """A question label's value for one row, joined through the ground truth
    a row's `id` names — a run file's rows carry no label of their own."""
    question = questions.get(row.get('id')) or {}
    return (question.get('question_metadata') or {}).get(name)


def difficulty_rates(rows: list[dict], questions: dict) -> list[dict]:
    """One row per value of a 'difficulty' question label, in the order the
    ground truth's own values first appear — a corpus that declares none
    names no bands, rather than one invented for it. `evidence_found` and
    `quotes_in_context` stay separate from `answered`, since retrieval
    reaching the evidence and the answer using it are different failures; both
    are `None` where the band holds only unanswerable questions."""
    out = []
    for name in dict.fromkeys(_label(questions, row, 'difficulty')
                              for row in rows):
        if name is None:
            continue
        group = [row for row in rows if _label(questions, row, 'difficulty') == name]
        answerable = [row for row in group if row.get('behavior') != 'abstain']
        out.append({
            'difficulty': name,
            'n': len(group),
            'n_answerable': len(answerable),
            'answered': round(
                sum(1 for row in group if answered_correctly(row)) / len(group), 4),
            'evidence_found': _numeric_mean([row.get('hit') for row in answerable]),
            'quotes_in_context': (
                round(sum(1 for row in answerable
                          if row.get('quote_recall') == 1.0) / len(answerable), 4)
                if answerable else None),
            'recall': _numeric_mean([row.get('recall') for row in answerable]),
            'answer_overlap': _numeric_mean([row.get('answer_token_f1')
                                             for row in group]),
        })
    return out


def type_rates(rows: list[dict], questions: dict) -> list[dict]:
    """The same table by a 'question_type' question label: an aggregate hides
    a change that helps one kind of question and hurts another."""
    out = []
    for name in dict.fromkeys(_label(questions, row, 'question_type')
                              for row in rows):
        if name is None:
            continue
        group = [row for row in rows
                 if _label(questions, row, 'question_type') == name]
        out.append({
            'type': name, 'n': len(group),
            'answered': round(
                sum(1 for row in group if answered_correctly(row)) / len(group), 4),
            'recall': _numeric_mean([row.get('recall') for row in group]),
        })
    return out


def _table(header: list[str], body: list[list[str]]) -> str:
    lines = ['| ' + ' | '.join(header) + ' |',
             '| ' + ' | '.join('---' for _ in header) + ' |']
    lines += ['| ' + ' | '.join(cell for cell in row) + ' |' for row in body]
    return '\n'.join(lines)


def question_page(run: dict, question: dict, row: dict) -> str:
    """One question: what was asked, what the truth is, what came back, what it
    replied, and what graded it."""
    gold = {str(relevant['corpus_document_id'])
            for relevant in question.get('relevant_corpus_documents') or []}
    measures = {measure.key: measure for measure in metrics.MEASURES}
    question_metadata = question.get('question_metadata') or {}
    expected = question['expected_answer']
    behavior = expected['behavior']
    parts = [
        f"# {question['groundtruth_question_id']} — "
        f"{question_metadata.get('question_type', '—')} / "
        f"{question_metadata.get('difficulty', '—')}",
        '',
        f"Run `{run['run_id']}` · **{run.get('label', '')}** · "
        f"k={run['config']['retrieval'].get('k')} · "
        f"chunker `{run['config']['index'].get('chunker')}` · "
        f"answerer `{run['config']['generation'].get('answerer')}`",
        '',
        '## Asked',
        '',
        f"> {question['question']}",
        '',
        f"Behavior: **{behavior}** · "
        f"declared time scope {question_metadata.get('resolved_time_scope') or '—'} · "
        f"detected time scope {row.get('time_scope') or '—'}",
        '',
        '## Reference',
        '',
    ]
    if expected.get('text'):
        parts += [f"> {expected['text']}", '']
    if expected.get('derived_facts'):
        parts += ['Derived facts the answer has to contain:', '']
        parts += [f'- {fact["fact"]}' for fact in expected['derived_facts']] + ['']
    if question.get('relevant_corpus_documents'):
        parts += ['Evidence, with what the ground truth cites:', '']
        parts += [f"- `{relevant['corpus_document_id']}` [{ev.get('fidelity')}] "
                  f"— «{ev['text']}»"
                  for relevant in question['relevant_corpus_documents']
                  for ev in relevant.get('evidence') or []] + ['']
    else:
        parts += ['No evidence: this question is unanswerable from the corpus, '
                  'and the correct response is a refusal.', '']

    retrieved = row.get('retrieved_sessions') or []
    parts += [
        '## Retrieved',
        '',
        f"{row.get('n_contexts', '—')} chunks · "
        f"{row.get('context_chars', '—')} characters · "
        f"{_num(row.get('latency_ms'), 1)} ms",
        '',
    ]
    if retrieved:
        parts += ['Sessions in rank order — ✓ marks one the ground truth cites:',
                  '']
        parts += [f"{i}. `{session}`{'  ✓' if session in gold else ''}"
                  for i, session in enumerate(retrieved, start=1)]
    else:
        parts += ['No session-bearing chunk was retrieved. Chunks carry '
                  'no session id, so a ledger or a digest can be the whole '
                  'context and leave this list empty.']
    parts += ['',
              '*The chunk text is not stored in a run file — only the session '
              'ids above. Reconstructing the chunks would mean re-running '
              'retrieval, which would document a different retrieval than the '
              'one that was graded.*',
              '',
              '## Answered',
              '']
    parts += [f"Refused: **{'yes' if row.get('abstained') else 'no'}**", '']
    parts += ['```', (row.get('answer') or '(no answer)').strip(), '```', '']

    parts += ['## Graded', '']
    body = []
    for key in GRADE_KEYS:
        if row.get(key) is None:
            continue
        measure = measures.get(key)
        body.append([
            measure.label if measure else key,
            f'`{key}`',
            _num(row[key]),
            f'`{measure.formula}`' if measure else '—',
            measure.library if measure else '—',
        ])
    body.append(['Answered from the right evidence', '`answered`',
                 'yes' if answered_correctly(row) else 'no',
                 '`(answerable ∧ ¬refused ∧ hit) ∨ (¬answerable ∧ refused)`',
                 'raglab.agents.extra_tools.export.answered_correctly (no model)'])
    parts += [_table(['Grade', 'Key', 'Value', 'Formula', 'Computed by'], body), '']

    judged = (run.get('ragas') or {}).get('metrics') or {}
    if judged:
        parts += [
            '### The judged metrics are run means, not per question',
            '',
            'RAGAS scores every sample and the lab stores the averages, so the '
            'four numbers that chose this architecture cannot be attributed to '
            'this question. They are the whole run:',
            '',
        ]
        parts += [f'- `{key}` = {_num(value)} (run mean over '
                  f"{(run.get('ragas') or {}).get('n_samples', '?')} samples)"
                  for key, value in sorted(judged.items())]
        parts += ['']
    return '\n'.join(parts) + '\n'


def index_page(run: dict, rows: list[dict], questions: dict) -> str:
    """The folder's own front page: how to read it, and the two rate tables.
    `questions` joins a row's bare `id` back to its `question_metadata`,
    since a run file's own rows carry no label — the two rate tables read
    whichever question labels the ground truth happens to declare as
    'difficulty' and 'question_type' and stay empty on a corpus that
    declares neither, rather than assuming every corpus shares them (D7)."""
    spread = (run.get('ragas') or {}).get('decision_spread') or {}
    decision = (run.get('ragas') or {}).get('decision')
    parts = [
        '# Per-question record of the chosen configuration',
        '',
        f"Run `{run['run_id']}` · **{run.get('label', '')}** · "
        f"{len(rows)} questions · {run.get('seconds')}s",
        '',
        f"RAGAS decision score **{_num(decision)}**"
        + (f" ± {_num(spread.get('stderr'))} (standard error over "
           f"{spread.get('n')} questions)" if spread.get('stderr') is not None
           else ' (no measured spread)'),
        '',
        'One file per question, each showing what was asked, what the ground '
        'truth says, which documents came back, what the pipeline replied, and '
        'every grade with its own arithmetic beside it.',
        '',
        '## What "answered" means here',
        '',
        'A run file stores no judged grade per question, so this column is '
        'evidence-based and deliberately claims less than "correct": an '
        'answerable question counts when the pipeline did **not** refuse and '
        'retrieval reached a document the ground truth cites; an unanswerable '
        'one counts when it **did** refuse. Inventing an answer that was not '
        'there and refusing one that was are the two failures that matter, and '
        'both are counted. Whether the wording of a non-refused answer is right '
        'is graded separately, by the four judged metrics — as run means only.',
        '',
        '## By difficulty',
        '',
        _table(['Difficulty', 'n', 'Answered', 'Evidence found',
                'Gold quote in context', 'Recall@k', 'Answer overlap'],
               [[row['difficulty'], str(row['n']),
                 f"**{row['answered'] * 100:.0f}%**",
                 _num(row['evidence_found']), _num(row['quotes_in_context']),
                 _num(row['recall']), _num(row['answer_overlap'])]
                for row in difficulty_rates(rows, questions)]),
        '',
        '## By question type',
        '',
        _table(['Type', 'n', 'Answered', 'Recall@k'],
               [[row['type'], str(row['n']), f"{row['answered'] * 100:.0f}%",
                 _num(row['recall'])] for row in type_rates(rows, questions)]),
        '',
        '## Questions',
        '',
    ]
    for row in rows:
        mark = '✓' if answered_correctly(row) else '·'
        parts.append(f"- {mark} [{row['id']}]({row['id']}.md) — "
                     f"{_label(questions, row, 'question_type') or ''} / "
                     f"{_label(questions, row, 'difficulty') or ''}")
    return '\n'.join(parts) + '\n'


def write_run(run: dict, ground_truth: dict, out_dir) -> list[Path]:
    """Write the index and one page per question. Returns the paths written."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    questions = {q['groundtruth_question_id']: q
                for q in ground_truth['groundtruth_dataset']}
    rows = [row for row in run['rows'] if row['id'] in questions]
    written = [out / 'README.md']
    written[0].write_text(index_page(run, rows, questions), encoding='utf-8')
    for row in rows:
        path = out / f"{row['id']}.md"
        path.write_text(question_page(run, questions[row['id']], row),
                        encoding='utf-8')
        written.append(path)
    return written
