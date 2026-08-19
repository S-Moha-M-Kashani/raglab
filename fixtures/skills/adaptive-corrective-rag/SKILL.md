---
name: adaptive-corrective-rag
description: Decide per query whether to retrieve at all, which pipeline to route it to, and what to do when the retrieved evidence is bad — Self-RAG's reflection tokens, CRAG's retrieval evaluator and correction paths, Adaptive-RAG's complexity router. Use when one fixed pipeline is either too expensive for easy questions or too weak for hard ones, or when bad retrieval currently flows straight into generation unchallenged. Covers the measured routing benchmarks, why a cheap classifier often beats an LLM router, and the failure mode where an unreachable grader turns a gate into a no-op nothing reports.
---

# Adaptive and Corrective RAG

**What it is.** Two related admissions that a fixed `retrieve → generate`
pipeline is wrong for most queries:

- **Adaptive** — different questions deserve different pipelines. Some need no
  retrieval at all; some need one hop; some need many. Running the expensive path
  for all of them wastes most of the budget, and running the cheap path for all of
  them fails the hard ones.
- **Corrective** — retrieval sometimes returns rubbish, and a pipeline that feeds
  rubbish to the generator has chosen to hallucinate. Check the evidence *before*
  generating, and have somewhere to go when the check fails.

## Self-RAG

Trains reflection tokens into the model itself, so the model emits control
decisions inline as it generates: whether to retrieve now, whether a retrieved
passage is relevant, whether the generated sentence is supported by it, and
whether the output is useful. Retrieval becomes on-demand and per-segment rather
than once up front, and generation is self-critiqued as it goes.

The cost is that it is a *training* method — the reflection tokens have to be in
the model. That makes it a fine-tuning project rather than a pipeline knob, which
is why most production systems implement its spirit with prompted graders
instead.

## CRAG — Corrective RAG

A lightweight retrieval evaluator scores the retrieved documents and sorts the
query into one of three states:

- **Correct** — evidence is good. Refine it (decompose into strips, drop the
  irrelevant ones, recompose) and generate.
- **Incorrect** — evidence is bad. Discard it entirely and fall back to another
  source, classically a web search with a rewritten query.
- **Ambiguous** — take both paths and combine.

Measured on text-and-table documents, CRAG reached **Recall@5 of 0.658**, above
BM25, with **63% of queries triggering the correction path** — the correction is
not a rare edge case, it is most of the traffic.

The important design point is the *decompose-recompose* step: even in the
"correct" branch, CRAG does not pass retrieval through untouched. It strips the
irrelevant parts out, because a relevant document is usually mostly irrelevant
text.

## Adaptive-RAG and query routing

Train a classifier on query complexity and route to one of three strategies: no
retrieval, single-step retrieval, multi-step retrieval. The original uses a
T5-Large classifier trained on labels derived automatically from which strategy
actually succeeded — the labels are outcomes, not human judgement, which is what
makes the approach cheap to build.

The result: a three-class router **matches always-expensive baselines at
substantially lower cost**. That is the entire promise — not better answers, the
same answers for less.

### How well routing actually works, as of 2026

`RAGRouter-Bench` is the first dedicated benchmark for adaptive RAG routing, and
its numbers are a useful corrective to the enthusiasm:

| Router | Correct rate |
| --- | --- |
| Random | 39.06% |
| Best fixed strategy | 39.32% |
| Best learned router | 43.69% |
| Oracle | 60.83% |

The learned router beats both baselines, and the gap to oracle is larger than the
gap it closed. Current models do not capture query–corpus compatibility well.
Routing is worth doing and is not solved.

A second 2026 result cuts against reaching for an LLM to do it: a **TF-IDF plus
SVM** router reached macro-F1 0.928 and 93.2% accuracy while simulating **28.1%
token savings**. If the router itself needs a model call, it has eaten a good
part of the saving it exists to produce.

## The failure mode that matters most

A corrective gate is a filter with a threshold, and filters fail open.

If the grader cannot reach its model and returns a neutral score — 0.5 on a 0-to-1
scale — and the threshold is 0.4, then *every document passes*. The gate is a
no-op. The run completes, the numbers look normal, and every field on the row
still says the gate was on. You have measured the ungated pipeline and labelled it
gated.

The rule that prevents it: **an unreadable or unavailable verdict is read in the
direction that costs work**, and a grader that cannot reach its model must refuse
the run rather than score it. Distinguish two cases carefully:

- *The model answered and we could not parse it* — genuinely no opinion about one
  document among many. A neutral score is defensible.
- *The model was never reached* — no measurement happened. This must raise.

## When it pays and when it does not

Adaptive routing pays when query difficulty varies a lot and the expensive path
is genuinely expensive — a multi-hop agent loop, a paid judge, a web search.
It does not pay when all queries are similar, or when the router costs as much as
the path it is choosing between.

Correction pays when retrieval failure is common and has somewhere to go. If
there is no fallback source, "the evidence is bad" can only produce an abstention
— still valuable, but a much smaller win than the papers report, since theirs
comes largely from the web-search branch.

## In this lab

- **Candidate F is a corrective gate**, and it is the chosen architecture:
  `grader='llm'` with `grade_threshold=0.4` between retrieval and generation. It
  is the only candidate that changes anything *after* retrieval, which is why it
  was chosen — two of the four deciding metrics live on that side.
- **The 2026-08-02 fault is the failure mode above, measured**: with the daemon
  down, `retrieval.llm_scores` caught every exception and returned 0.5 per
  document, which cleared the 0.4 threshold. Ungated kept 4 contexts;
  `grader='llm'` kept 4 contexts. It now raises `GradeUnavailable`, which surfaces
  as a job in state `error` naming the stage. The parse fallback stays, because
  an unparsable line is genuinely no opinion.
- **F's own row is a statistical tie with the baseline** —
  0.7375 ± 0.0333 against 0.7222 ± 0.0341 on 30 questions. The architecture
  stands on the reasoning, not on the score, and the record says so.
- **No router exists here.** Every question takes the same pipeline. Adding one
  would be a genuine new candidate, and the cheap-classifier result above says
  the first version should not be an LLM call — the lab's `difficulty` labels on
  the ground truth are ready-made training targets for exactly the outcome-derived
  labelling Adaptive-RAG uses.

## Sources

- [RAGRouter-Bench: A Dataset and Benchmark for Adaptive RAG Routing](https://arxiv.org/abs/2602.00296)
- [Lightweight Query Routing for Adaptive RAG: A Baseline Study on RAGRouter-Bench](https://arxiv.org/abs/2604.03455)
- [From BM25 to Corrective RAG: Benchmarking Retrieval Strategies for Text-and-Table Documents](https://arxiv.org/html/2604.01733v1)
- [Agentic RAG Explained: Self-Correcting Retrieval](https://letsdatascience.com/blog/agentic-rag-self-correcting-retrieval)
