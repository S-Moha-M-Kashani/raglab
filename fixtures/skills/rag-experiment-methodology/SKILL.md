---
name: rag-experiment-methodology
description: Run RAG development as a disciplined experimental loop — baseline first, error analysis before fixes, one change per candidate, a dev set you iterate on and a held-out test set you almost never touch — instead of tuning by impression or chasing 100% on the set you tune against. Use when starting a new RAG project, when deciding how to split ground truth, when someone proposes cross-validation or driving the dev set to perfection, and before reporting any final number. Covers the config-selection view of RAG training, the failure taxonomy that localises an error to a stage, why 100% on the dev set is a bug and not a goal, when cross-validation pays and when it cannot, and test-set hygiene.
---

# RAG Experiment Methodology

**What it is.** The industrially and scientifically sound loop for improving a
RAG system. It is the same methodology classical ML settled on decades ago —
baseline, error analysis, hypothesis, held-out evaluation — with two
adjustments RAG forces: "training" is **configuration selection**, not gradient
descent, and the most informative evaluation signal is **judged**, which makes
every evaluation cost money and changes what is affordable.

## What "training" means when nothing has gradients

A RAG pipeline is mostly frozen models joined by knobs. Improving it means
choosing chunker, k, reranker, gate, prompts — a search over configurations,
guided by a labelled set. That set plays exactly the role a training set plays:
**whatever you iterate against, you will overfit to.** Not through weights —
through your choices. Thirty questions will reward the config that happens to
suit those thirty.

So the classical split survives with new names:

- **Dev set** — the questions you run error analysis on, look at freely, debug
  against, and select candidates with. This is where you live.
- **Test set** — held out, balanced the same way, touched only to compare the
  final few candidates. Its numbers are the ones you report.
- **Judge-calibration set** — if an LLM judge decides anything, it is itself an
  instrument that needs its own labelled split (roughly 10–20% to build the
  judge's prompt, 40–45% to iterate on it, 40–45% held out to measure it —
  Hamel Husain's split), *separate from* the pipeline's dev/test. A judge tuned
  on the same questions it later grades is contaminated.

## The loop

1. **Validate the dataset first.** Refuse malformed ground truth rather than
   repairing it silently; a corpus that misquotes itself scores confidently
   about text that was never there.
2. **Screen the judge** before looking at any leaderboard it would produce —
   otherwise selecting a judge is judge-shopping (see `rag-evaluation`).
3. **Establish the baseline row.** The simplest defensible pipeline (recursive
   chunks, hybrid retrieval, no rerank, no gate), measured on the dev set with
   error bars. Every later candidate is read against it.
4. **Error analysis before any fix.** Read the failing traces by hand — 20–50
   outputs per pass, and keep reading until new traces stop revealing new
   failure modes (theoretical saturation; ~100 traces is the working target).
   Classify each failure; do not fix anything yet.
5. **One hypothesis → one knob → one candidate.** The dominant failure class
   names the lever. A candidate differing in three knobs cannot attribute its
   win to any of them.
6. **Compare with error bars, on the same questions, under the same judge.**
   Inside the combined error is a tie, and a tie means the knob does not
   matter — which is a finding, not a failure.
7. **Stop when the residue is noise.** The loop ends when remaining dev
   failures are label errors, ambiguous questions, or irreducible model limits
   — **not when the dev score reaches 100%** (below).
8. **Then, once: the test set.** The final one to three candidates, one run
   each, error bars, report. What you learn from test failures feeds the *next
   cycle's* dev work; it does not license another test run this cycle.

## Why 100% on the dev set is a bug, not a goal

Driving the tuning set to perfection feels rigorous and is the opposite:

- **The last points are bought with overfitting.** Past the honest plateau,
  every further gain comes from encoding dev-set idiosyncrasies into the
  config. The dev score rises as the test score falls — the textbook curve.
- **Ground truth contains errors**, typically a few percent. Reaching 100%
  *requires* fitting those errors — building a pipeline that reproduces the
  dataset's bugs.
- **It destroys the instrument.** A dev set the pipeline aces can no longer
  rank candidates; you have spent its resolution.

The right target: **explain 100%, not score 100%.** Every dev failure gets a
diagnosis — label error, retrieval miss, generation failure, judge error — and
the loop stops when the diagnoses stop naming fixable causes. When the
diagnosis is a dataset bug, fix the dataset — but **version it**: rows measured
against the old ground truth are incomparable with rows against the new, and a
leaderboard must group by question set rather than pretend continuity.

## The failure taxonomy, which is the whole point of a pipeline

"The answer was wrong" is not a finding. A RAG failure localises to a stage,
and each stage has a different fix and a different owner:

| # | Failure | Diagnosed by | Fix lives in |
| --- | --- | --- | --- |
| 0 | answer not in corpus / label error | reading the source | the dataset |
| 1 | retrieval miss — gold not in candidates | recall@depth | chunking, embedder, query transform |
| 2 | rank miss — in candidates, not in top-k | rank trace per step | reranker, k, fusion |
| 3 | gate drop — retrieved, then filtered out | gate's per-doc scores | threshold, grader |
| 4 | generation failure — evidence present, answer wrong | reading answer vs context | prompt, model, context budget |
| 5 | judge error — answer right, scored wrong | human spot-check | the judge, not the pipeline |
| 6 | infrastructure — model unreachable, empty reply | stated errors on the row | never a tuning decision |

Class 6 is why silent fallbacks are forbidden (`rag-evaluation`): an
infrastructure failure scored as a quality failure sends the loop off fixing
the wrong thing. And the taxonomy is why per-stage traces are worth their cost
— without a rank-per-step trace, classes 1–3 are indistinguishable.

## Cross-validation: when it pays, when it cannot

CV exists to stabilise estimates on small data, and question sets here are
small — so it looks natural. The constraint is cost asymmetry:

- **Deterministic metrics** (recall, nDCG, MRR): re-scoring is nearly free once
  retrieval ran, so k-fold or bootstrap over questions costs little and
  stabilises retrieval comparisons. Use it.
- **Judged metrics**: every fold is a full paid judged run — k-fold multiplies
  hours and spend by k. The affordable substitute is one fixed,
  difficulty-balanced dev set with **per-question standard errors**, which is a
  bootstrap-flavoured answer to the same question: is this gap real?

Balance any split by difficulty and question type; a natural distribution is
skewed, and a plain stride measures one band and reports it as the pipeline.
And never re-split between candidates — comparability requires the same
questions.

## In this lab

Most of this loop is load-bearing structure here already: the sweep's
one-knob-per-candidate rule, `decision_spread()`'s per-question error,
`leaderboard.group`'s same-questions-same-judge rule and its `tie` verdict,
`judgescreen` run before the leaderboard, `datasets.validate()` refusing rather
than repairing, balanced sampling with remainder rules, and the Inspector's
per-step rank trace — which is the failure taxonomy's classes 1–3 made
readable. The stated stop reasons (`answer_error`, `GradeUnavailable`) are
class 6 kept out of the quality signal.

**The honest gap: the lab has no held-out test set.** `SWEEP_LIMIT`'s 30
balanced questions are both the selection set and the report set — the
leaderboard's numbers are dev numbers, and the winning candidate's score
carries selection bias (the winner of many comparisons is flattered by its
luck on those specific 30). The fix is cheap and fits the existing design:
freeze a disjoint, difficulty-balanced holdout from the remaining questions,
run only finalists there, and let the leaderboard's grouping rule keep the two
sets in separate tables, which it already would. Candidate F's tie with the
baseline (0.7375 ± 0.0333 vs 0.7222 ± 0.0341) was read correctly as a tie —
the discipline held — but the number itself is still a dev-set number.

## Sources

- [LLM Evals: Everything You Need to Know — Hamel Husain](https://hamel.dev/blog/posts/evals-faq/)
- [Evals: Doing Error Analysis Before Writing Tests — Hamel Husain](https://hamel.dev/notes/llm/officehours/erroranalysis.html)
- [A pragmatic guide to LLM evals for devs — The Pragmatic Engineer](https://newsletter.pragmaticengineer.com/p/evals)
- [Building eval systems that improve your AI product — Lenny's Newsletter](https://www.lennysnewsletter.com/p/building-eval-systems-that-improve)
- [Notes on LLM Evaluation — Towards Data Science](https://towardsdatascience.com/notes-on-llm-evaluation/)
