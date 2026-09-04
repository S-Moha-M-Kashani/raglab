"""Sweep candidate architectures and rank them on the four deciding metrics; drives `evaluate.run_eval` and writes to the same `.runs/` as the panel.
Every candidate changes exactly one thing against the baseline, so a win can be attributed to it; `RAGLAB_LLM` picks the answerer/judge pair too (`PAIRINGS`).
Screen the judge first (`uv run raglab-judgescreen`) — an unscreened judge can score every candidate identically and tied, silently.
"""
import argparse
import json
import os
import sys
import time
from dataclasses import replace

from raglab.evaluation import leaderboard
from raglab.llm_backends import cli_subprocess_chat as clichat
from raglab.corpora import corpus_reading as corpus
from raglab.corpora import dataset_import_contract as datasets
from raglab.evaluation import ragas_judged_metrics as ragas_eval
from raglab.configuration import split_plan
from raglab.configuration.lab_config import (
    GenerationConfig,
    IndexConfig,
    LabConfig,
    RetrievalConfig,
    RUNS_DIR,
    load_lab_settings)
from raglab.evaluation.run_evaluation import run_eval
from raglab.rag_components.indexing.index_builder_registry import IndexRegistry

# Held fixed across every candidate: varying it alongside the knobs would make
# each row differ in two things.
EMBEDDER = 'sentence-transformers'
EMBED_MODEL = 'heydariAI/persian-embeddings'

# The answerer and the judge are deliberately different models — a model grading
# its own output is not evidence. One pairing per provider, since a slug only
# means something to the backend that serves it.
PAIRINGS = {'openrouter': {'answerer': 'openai/gpt-5-nano',
                           'judge': 'openai/gpt-5-mini'},
            'ollama': {'answerer': '4skl/gemma4-e2b-mtp',
                       'judge': 'gemma4:e2b'},
            # Codex is deliberately absent: only one alias is verified here, so
            # there is no honest pair — a sweep there must name both by hand.
            'claude': {'answerer': 'sonnet',
                       'judge': 'opus'},
            'fake': {'answerer': 'openai/gpt-5-nano',
                     'judge': 'openai/gpt-5-mini'}}

# Read at import so the env that chooses the backend also chooses the pairing.
# No fallback to another backend's pins: that would hand a CLI backend a slug
# it has never heard of — a backend with no pinned pair cannot sweep.
_PROVIDER = os.environ.get('RAGLAB_LLM', '') or 'openrouter'
_PAIR = PAIRINGS.get(_PROVIDER, {})
ANSWER_MODEL = _PAIR.get('answerer', '')
JUDGE_MODEL = _PAIR.get('judge', '')

# Every candidate is measured on the same 30 questions — 10 easy, 10 medium, 10
# hard — rather than the full 112, since a candidate sweep pays the judged cost
# per row and a skewed sample would measure one band's pipeline as the pipeline.
SWEEP_LIMIT = 30
SWEEP_BALANCE = 'difficulty'

BASE = LabConfig(
    index=IndexConfig(embedder=EMBEDDER, embed_model=EMBED_MODEL),
    retrieval=RetrievalConfig(),
    # answerer='llm': the four deciding metrics all need a generated response.
    generation=GenerationConfig(answerer='llm', model=ANSWER_MODEL,
                                ragas_model=JUDGE_MODEL),
    label='A baseline')


def candidates() -> list[LabConfig]:
    """One hypothesis per row, each a single change against the baseline."""
    out = [BASE]

    def variant(label, *, index=None, retrieval=None):
        cfg = BASE
        if index:
            cfg = replace(cfg, index=replace(cfg.index, **index))
        if retrieval:
            cfg = replace(cfg, retrieval=replace(cfg.retrieval, **retrieval))
        out.append(replace(cfg, label=label))

    variant('C tighter context k=5', retrieval={'k': 5})
    variant('D wider context k=12', retrieval={'k': 12})
    variant('F llm relevance gate', retrieval={'grader': 'llm',
                                               'grade_threshold': 0.4,
                                               'grader_model': ANSWER_MODEL})
    variant('H whole-document chunks',
            index={'split_plan': split_plan.parse('document')})
    return out


def score(result) -> float | None:
    return (result.ragas or {}).get('decision')


def line(label: str, result) -> str:
    metrics = (result.ragas or {}).get('metrics', {})
    overall = result.summary.get('overall', {})
    parts = ' '.join(
        f'{name.split("_")[0][:5]}={metrics.get(name)!s:>6}'
        for name in ragas_eval.DECISION_METRICS)
    return (f'{label:24} decision={str(score(result)):>7}  {parts}  '
            f'headline={overall.get("headline")} '
            f'recall={overall.get("recall")} '
            f'quote={overall.get("quote_recall")}  {result.seconds}s')


def judged_settings():
    """Settings, or exit — every run here is ranked on judged metrics, and with
    no backend the fake provider would answer and judge without failing. Refuses
    on whether a *real model* can be reached, not on whether a credential exists:
    a judge served by Ollama needs no key."""
    settings = load_lab_settings()
    if not settings.llm_ready:
        sys.exit('no LLM backend: the four deciding metrics are judged, so there '
                 'is nothing to rank without one. Four ways out: '
                 'RAGLAB_LLM=claude and RAGLAB_LLM=codex run a CLI already '
                 'installed and logged in on this machine and need no key at '
                 'all; RAGLAB_LLM=ollama runs a model on it; '
                 'RAGLAB_LLM=openrouter needs OPENROUTER_API_KEY. Each has a '
                 'pinned answerer/judge pair except codex, where one alias is '
                 'verified here and a model grading its own output is not '
                 'evidence — sweep on a backend with a pinned pair.')
    if not ANSWER_MODEL or not JUDGE_MODEL:
        sys.exit(f'no answerer/judge pair for RAGLAB_LLM={_PROVIDER!r}: this '
                 'backend has no pinned pair in PAIRINGS, and a model grading '
                 'its own output is not evidence — sweep on a backend that '
                 'has one.')
    if ANSWER_MODEL == JUDGE_MODEL:
        sys.exit(f'answerer and judge are both {ANSWER_MODEL!r}: a model grading '
                 'its own output is not evidence, and these four metrics are the '
                 'whole basis of the ranking')
    return settings


def capped_workers(workers: int, settings) -> int:
    """How many questions may be answered at once, on this backend. On a CLI
    backend every LLM call in the answering phase is a whole process, so this
    reads the same cap `ragas_eval.JUDGE_LOAD` uses for the judging phase rather
    than inventing a second number. Never raises what was asked for — only lowers
    an unmeasured default, never overrules a deliberately lower one."""
    if settings.provider not in clichat.CLIS:
        return workers
    return min(workers, ragas_eval.judge_load(settings)['max_workers'])


BAR_WIDTH = 28


def bar(label: str, stage: str, fraction: float, detail: str,
        started: float) -> str:
    """One line, rewritten in place. Elapsed time is shown because the fraction
    alone cannot tell a slow phase from a stuck one."""
    filled = int(round(BAR_WIDTH * min(1.0, max(0.0, fraction))))
    mins, secs = divmod(int(time.time() - started), 60)
    tail = f' · {detail}' if detail else ''
    return (f'\r  {label[:18]:18} [{"█" * filled}{"·" * (BAR_WIDTH - filled)}] '
            f'{fraction * 100:5.1f}% {mins:>3}m{secs:02d}s  {stage}{tail}')


def live(label: str, started: float, stream=sys.stdout):
    """A progress callback that rewrites one terminal line, padded to a fixed
    width so a short detail does not leave the tail of a longer one behind it."""
    def report(stage: str, fraction: float, detail: str = '') -> None:
        stream.write(f'{bar(label, stage, fraction, detail, started):<118}')
        stream.flush()
    return report


def sweep(limit: int, workers: int, only: list[str] | None = None,
          balance: str = SWEEP_BALANCE) -> list[tuple]:
    settings = judged_settings()
    diary, ground_truth = datasets.load()
    registry = IndexRegistry(settings, diary)

    asked, workers = workers, capped_workers(workers, settings)
    picked = [c for c in candidates()
              if not only or c.label.split()[0] in only]
    print(f'{len(picked)} candidates · {limit} questions each ({balance}) · '
          f'{workers} workers · {settings.provider} · judge {JUDGE_MODEL} · '
          f'answerer {ANSWER_MODEL}')
    if workers != asked:
        print(f'  (asked for {asked}; a call here is a process, and this backend '
              f'is capped at {workers} by ragas_eval.JUDGE_LOAD)')
    per_row = ragas_eval.expected_judge_calls(limit, BASE.retrieval.k)
    print(f'~{per_row} judge calls per candidate, ~{per_row * len(picked)} in '
          f'total — an estimate: RAGAS retries malformed output\n')
    scored = []
    for i, cfg in enumerate(picked, start=1):
        started = time.time()
        stage = cfg.label.split()[0]
        print(f'[{i}/{len(picked)}] {cfg.label} …', flush=True)
        result = run_eval(registry, ground_truth, cfg, settings, limit=limit,
                          balance=balance, ragas_mode='llm', ragas_limit=limit,
                          workers=workers,
                          progress=live(f'Stage {stage}', started))
        print()                              # close the rewritten bar line
        scored.append((score(result), cfg.label, result))
        print('   ' + line(cfg.label, result))
        print(f'   run {result.run_id} · {round(time.time() - started)}s\n',
              flush=True)

    ranked = sorted(scored, key=lambda row: (row[0] is None, -(row[0] or 0)))
    print('\n=== ranked by the RAGAS decision score '
          f'({", ".join(ragas_eval.DECISION_METRICS)}) ===')
    for value, label, result in ranked:
        print('   ' + line(label, result))
    # A sorted list reads as a result; whether the top row actually beat the one
    # below it has to be answered here, not left to whoever reads the ordering.
    for note in ranking_verdict([replace_label(result, label)
                                 for _, label, result in ranked]):
        print(note)
    return ranked


def replace_label(result, label: str) -> dict:
    """A leaderboard row from a run, carrying the candidate's letter. `brief()`
    drops the question ids for rendering; they are put back from `selection`
    because the comparability grouping needs them."""
    row = result.brief()
    row['label'] = label
    row['selection'] = dict(result.selection)
    row['judge'] = (result.ragas or {}).get('judge') or {}
    return row


def ranking_verdict(rows: list[dict]) -> list[str]:
    """Whether the ordering above is a result, in words — using the same refusal
    terms as `leaderboard`, so a printed ranking cannot claim more than it measured."""
    out = []
    for found in leaderboard.group(rows):
        call = leaderboard.verdict(found)
        measured = [r for r in found.rows if r.get('ragas_decision') is not None]
        head = f'   {found.sample} · judged by {found.judge}'
        if call == 'tie':
            best, second = measured[0], measured[1]
            lead = best['ragas_decision'] - second['ragas_decision']
            errors = (best['ragas_decision_stderr'] ** 2
                      + second['ragas_decision_stderr'] ** 2) ** 0.5
            out.append(f'{head}\n   No winner: {best["label"]} leads '
                       f'{second["label"]} by {lead:.4f}, inside the combined '
                       f'error of {errors:.4f} — these rows do not separate.')
        elif call == 'unknown':
            out.append(f'{head}\n   No winner: no measured error on these rows, so '
                       'a lead cannot be told from noise.')
        elif call == 'unranked':
            out.append(f'{head}\n   One judged row only — nothing to compare it '
                       'against.')
        else:
            out.append(f'{head}\n   Winner: {call}, by more than the combined '
                       'error of the top two rows.')
    return out


def final(limit: int | None, workers: int, label: str,
          balance: str = SWEEP_BALANCE) -> None:
    """Re-run one candidate over the whole question set: the winner is decided
    on a subset for cost, but a per-type breakdown over two questions isn't one."""
    # Looked up before the backend check and the corpus load: a letter no
    # candidate carries is an argument error, not work worth a build.
    cfg = next((c for c in candidates() if c.label.split()[0] == label), None)
    if cfg is None:
        sys.exit(f'no candidate {label!r}: --final takes one candidate letter, '
                 'one of ' + ' '.join(c.label.split()[0] for c in candidates()))
    cfg = replace(cfg, label=f'WINNER {cfg.label} · full set')
    settings = judged_settings()
    diary, ground_truth = datasets.load()
    registry = IndexRegistry(settings, diary)
    workers = capped_workers(workers, settings)
    n = limit or len(ground_truth['groundtruth_dataset'])
    started = time.time()
    print(f'final run: {cfg.label} over {n} questions, {workers} workers',
          flush=True)
    result = run_eval(registry, ground_truth, cfg, settings, limit=limit,
                      balance=balance, ragas_mode='llm', ragas_limit=limit,
                      workers=workers,
                      progress=live(f'Final {label}', started))
    print()
    print(line(cfg.label, result))
    print(f'run {result.run_id}')
    print(json.dumps({'decision': score(result),
                      'ragas': (result.ragas or {}).get('metrics'),
                      'overall': result.summary.get('overall')},
                     ensure_ascii=False, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog='Example: uv run raglab-sweep --only A F --limit 10 --workers 3')
    parser.add_argument('--limit', type=int, default=SWEEP_LIMIT,
                        help='questions per candidate (default %(default)s, '
                             'balanced across the difficulty bands)')
    parser.add_argument('--balance', default=SWEEP_BALANCE,
                        help='the name of a question label to equalise '
                             '("difficulty" on a corpus that declares one), '
                             'or "" to stride the set as it is, which is what '
                             'the runs before 2026-07-31 used')
    parser.add_argument('--workers', type=int, default=6,
                        help='questions scored in parallel; the judged stages '
                             'are dominated by waiting on the model. Drop this '
                             'to 2–3 for a local model, which serves far fewer '
                             'concurrent requests than a remote API. On a CLI '
                             'backend it is capped from ragas_eval.JUDGE_LOAD, '
                             'because there every call is a whole process')
    parser.add_argument('--only', nargs='*',
                        help='candidate letters to run, e.g. --only A F')
    parser.add_argument('--final', metavar='LETTER',
                        help='re-run one candidate over the full question set')
    args = parser.parse_args()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if args.final:
        final(None, args.workers, args.final, args.balance)
    else:
        sweep(args.limit, args.workers, args.only, args.balance)


if __name__ == '__main__':
    main()
