---
name: rag-evaluation
description: Measure a RAG pipeline so the numbers can actually decide something — which metrics vote, why judged metrics beat deterministic ones for architecture choices, how to screen an LLM judge before trusting it, and how to report error so a ranking is not read as a win it did not earn. Use before running any comparison, before believing any leaderboard, and whenever someone proposes adding a technique because it improved a benchmark elsewhere. Covers RAGAS metrics, judge screening and degeneracy, one-knob ablation discipline, sampling balance, and the failure modes that produce confident numbers measuring nothing.
---

# RAG Evaluation

**What it is.** The part that decides whether any other skill in this collection
was worth adopting. Most RAG systems are tuned on impressions; the difference
between a workbench and a demo is whether a change can be shown to have helped.

## The 2×2 that determines which metrics you need

Errors happen on two axes: **which stage** (retrieval or generation) and **which
sin** (omission — missing what was needed; commission — including what was not).

| | Omission | Commission |
| --- | --- | --- |
| **Retrieval** | context recall | context precision |
| **Generation** | answer relevancy | faithfulness |

Those four cover the space. A metric set missing a cell cannot see a whole class
of failure — and a pipeline optimised against an incomplete set will drift into
the blind spot, because that is where the free wins are.

## Judged versus deterministic metrics

**Deterministic** — recall@k, precision@k, nDCG, MRR, exact match, token overlap.
Cheap, reproducible, zero variance. They measure whether the right chunk was
retrieved.

**Judged** — an LLM scores faithfulness, relevancy, groundedness. Expensive,
variable, and the only thing that measures whether the *answer* was any good.

The rule that follows: **judged metrics decide the architecture; deterministic
metrics are reported and do not vote.** Ranking on deterministic scores rewards a
pipeline that finds the evidence and then says nothing useful about it — and the
generation half of the 2×2 has no deterministic proxy worth trusting.

Then, having chosen the deciding set:

- **Take an unweighted mean.** A weighting is a claim about relative importance
  that a fixture cannot support, and a thumb on the scale is how a sweep confirms
  what its author already believed.
- **Require all of them.** A composite over whichever metrics happened to succeed
  lets the run that measured least win.

## Report the error, always

A decision score is a mean over questions. Means over 30 questions have standard
errors that are routinely larger than the difference between candidates.

Concretely: two configurations differing in one knob in opposite directions
scored 0.6487 and 0.6501 — precision and recall merely traded places. A bare
ranking reads that as a win. With error bars it is a tie, and the correct
conclusion is that the knob does not matter.

Compute the error over **per-question composites**, not as four independent
standard errors. The four judged metrics score the same answers and move
together, so treating them as independent understates the spread.

And show *nothing* rather than `± 0` when no error was measured — a zero
presents the oldest, least-instrumented rows as the most precise ones.

## Screening the judge

An LLM judge is a measuring instrument and needs calibrating before use. Two
failure modes, counted separately because they are fixed separately:

**Degeneracy.** The judge says yes to everything (or no to everything). On a
balanced set that scores 0.5 and separates no candidate from any other — a
perfectly useless instrument that produces perfectly normal-looking numbers. Two
local models have been measured answering identically to every claim.

**Schema adherence.** Structured-output frameworks retry on malformed responses.
A fast model that writes prose instead of JSON spends its entire speed advantage
on retries, and the retry count is a real cost that no metric reports.

**Build the screen from ground truth, never by hand.** The construction that
works: a supported claim is a question's own verified answer against its own
evidence; the unsupported claim is *that same answer with one numeral changed* to
one the context never states. Word-for-word identical apart from digits, so token
overlap is equal across the two classes — measured 0.533 against 0.517 — and a
judge secretly scoring word overlap cannot beat chance.

The first design of that screen used the most similar answer from a *different*
question as the negative, and was rejected on measurement: the true answer led on
overlap 0.43 to 0.20, so overlap alone would have passed the screen.

**Run the screen before looking at the leaderboard the judge would produce.**
Otherwise it is judge-shopping, and with a candidate field spanning ~0.01 that is
effortless to do by accident.

**A model must never grade its own output.** Refuse the pairing in code rather
than trusting the operator to remember.

## Comparison discipline

**One knob per candidate.** A row differing in three settings cannot attribute
its win to any of them. Hold the embedder and the models fixed across every
candidate.

**Same questions, same judge, or no comparison.** A decision score is a mean over
specific questions judged by a specific model. Group before ranking, one table
per (question set, judge) pair, and never rank across tables. Save the chosen
question ids on every run — two runs of 24 questions may be two different 24.

**Balance the sample by difficulty.** Natural distributions are skewed; a plain
stride over a 29/57/26 easy/medium/hard corpus hands medium about half the
sample, so you have measured one band and reported it as the pipeline. Take equal
quotas and, when the limit does not divide, give the remainder to the earlier
bands rather than silently shrinking the sample.

**Never change a sampling default underneath an existing leaderboard.** Old rows
become incomparable rather than merely old.

## Failure modes that produce confident numbers measuring nothing

The recurring shape: a component fails, something neutral is substituted, the run
completes, and no field on the row contradicts the label.

- **A fake or offline provider** answering and judging without ever failing — a
  full set of confident numbers from a pipeline that measured nothing. Record the
  resolved backend on every row, and refuse to start a sweep on it.
- **A neutral score clearing its own threshold** — an unreachable grader
  returning 0.5 against a 0.4 bar makes the gate a no-op. Read an unparsable
  verdict in the direction that costs work, never the one that passes.
- **A silent fallback** — a reranker that cannot load and passes the pre-rerank
  order through, still labelled as reranked.
- **A degenerate judge** — see above.
- **An empty reply read as an answer** — some models exit successfully and say
  nothing; an unparsed line then scores as neutral rather than as an error.
- **A refusal counted as a bad answer** — "refused because the corpus is silent"
  and "refused because the model was unreachable" score identically low on
  faithfulness unless the reason is on the row.

Every one of these is fixed the same way: **the row must state what actually
ran** — the resolved backend, the model, the stop reason, the error — and a
component that cannot do its job must refuse rather than substitute.

## In this lab

- `ragas_eval.DECISION_METRICS` is exactly the 2×2 above: faithfulness, answer
  relevancy, LLM context precision, context recall. `decision_score()` is their
  unweighted mean and is `None` unless all four are present.
- Everything else — `factual_correctness`, the offline context pair, all sixteen
  deterministic scores — is reported and does not vote.
- `decision_spread()` is the standard error over per-question composites, and the
  0.6487 / 0.6501 pair above is a real row from the 2026-07-30 sweep.
- `metrics.MEASURES` and `RAGAS_MEASURES` share one `Measure` shape: label, step,
  an arithmetic `formula`, and a `library` that names the module or RAGAS class
  *and* says whether a model was involved. `explain.missing_metrics() == []` is
  the gate that stops a metric shipping as a bare number.
- `llm_tools/judgescreen.py` is the screen described above, with the numeral-swap
  construction and both counted failure modes. `.screens/` is machine-local and
  git-ignored, and is the evidence for which model was allowed to grade.
- `llm_tools/leaderboard.py` groups before it ranks — by dataset first, then by
  (question set, judge). `verdict()` says `tie` inside the combined error and
  `unknown` when no error was measured; pre-`selection` runs key on question
  *count*, refuse a verdict, and carry no rank numbers, because a numbered row is
  a rank claim.
- `sweep.py` enforces one knob per candidate, holds the embedder and both models
  fixed, and `judged_settings()` refuses an answerer/judge pairing on the same
  model. Both entry points refuse to start on the `fake` backend.
- `SWEEP_LIMIT` / `SWEEP_BALANCE` take 10 easy, 10 medium, 10 hard; `_quotas`
  gives an indivisible remainder to the earlier bands. `balance='stride'` remains
  the default in `select_questions` because the existing `.runs/` rows were
  strided.
- `RunResult.selection` saves the chosen question ids and the sampling rule on
  every run.
- **Candidate F's own row is a tie**: 0.7375 ± 0.0333 against the baseline's
  0.7222 ± 0.0341 on 30 questions. It is the chosen architecture on the reasoning,
  and the record says so rather than implying a score it did not earn. That is
  what this whole apparatus is for.

## Sources

- [ARAGOG: Advanced RAG Output Grading](https://arxiv.org/pdf/2404.01037)
- [RAGSmith: Finding the Optimal Composition of RAG Methods Across Datasets](https://arxiv.org/pdf/2511.01386)
- [A Systematic Review of Key RAG Systems: Progress, Gaps, and Future Directions](https://arxiv.org/pdf/2507.18910)
- [RAGRouter-Bench](https://arxiv.org/abs/2602.00296) (for how far routing is from oracle, and why benchmarks need an oracle row)
