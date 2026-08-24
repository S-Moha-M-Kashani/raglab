---
name: rag-source-watchlist
description: The named sources to search for new RAG information, organised by use case — for each use case a project serves (conversational or agent memory, multilingual corpora, judged evaluation, and the always-on concerns of dependencies and local models), the links to check, the query phrases that find that literature, and which part of this repository a finding lands in. Use when looking for new work relevant to a specific use case rather than the field at large, when a model catalogue or dependency needs updating, or when a published benchmark matching a corpus shape is wanted. The generic sweep procedure and field-wide venues live in rag-research-radar; this skill is where that procedure points, per use case.
---

# RAG Source Watchlist

**What it is.** `rag-research-radar` answers "where does RAG research land and
how is it triaged" — field-wide, use-case-agnostic. This skill is the other
half: **per use case**, the sources that carry news about it, the query
phrases that find its literature, and where a finding lands in this
repository. The radar is the procedure; this is its address book. Watch the
sections whose use cases this installation currently serves; skim the rest.

## For the use case of conversational / agent memory

Timestamped multi-session chat plus questions requiring recall across sessions
— assistant memory, diaries, long-running support threads. The bundled default
corpus (`diary-fa`) has exactly this shape, which is why these
sources are first in the file, not because the lab is limited to them.

- [LongMemEval](https://arxiv.org/abs/2410.10813) ([code](https://github.com/xiaowu0162/longmemeval),
  [site](https://xiaowu0162.github.io/long-mem-eval/)) — ICLR 2025; ~500
  questions over timestamped chat histories testing **five abilities:
  information extraction, multi-session reasoning, temporal reasoning,
  knowledge updates, abstention**. For any memory-shaped corpus that list is a
  question-type checklist, and its headline finding — commercial assistants
  drop ~30% accuracy over sustained interaction — is the use case's problem
  statement.
- [LoCoMo](https://www.emergentmind.com/topics/locomo) — 50 conversations, up
  to 35 sessions, ~300 turns each; single-hop, multi-hop, temporal and
  open-domain questions over dialogue history.
- The memory-structured frontier in `rag-research-radar`'s snapshot (HGMem,
  xMemory, MemAdapter) — agent-memory systems whose retrieval half is this use
  case.

Query phrases: *conversational memory retrieval*, *multi-session temporal
reasoning QA*, *long-term memory chat assistant benchmark*, *lifelog / diary
question answering*; citation alerts on the two benchmarks.

## For the use case of a multilingual or non-English corpus

Any corpus whose language an English-tuned component can fail on silently
(see `multilingual-rag`). The bundled default is Farsi, which is why the lab
carries a measured example of the failure (the 60× encoder finding).

- [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) — filter
  by the corpus's language; where new encoders surface first.
- Language-specific MTEB variants where they exist —
  [FaMTEB](https://arxiv.org/pdf/2502.11571) for Persian is the model: the
  benchmark that should decide an encoder for that language, not a general
  leaderboard position.
- The default encoder's model page and its Hub neighbours
  ([heydariAI/persian-embeddings](https://huggingface.co/heydariAI/persian-embeddings)
  for the bundled corpus) — the candidate pool for the one drop-in upgrade
  always worth checking.
- Language-specific RAG best-practice lines where published —
  [Advancing RAG for Persian (LREC 2026)](https://aclanthology.org/2026.lrec-1.580/)
  is the bundled corpus's instance.

Query phrases: *{language} text embedding benchmark*, *{language} sentence
transformer*, *multilingual E5 successor*.

## For any use case that ranks candidates (judged evaluation)

Every use case, the moment two configurations are compared on judged metrics.

- [ragas documentation](https://docs.ragas.io/) and
  [ragas releases](https://github.com/explodinggradients/ragas/releases) —
  watched for two reasons: metric changes (a changed judge prompt moves every
  judged number, silently) and **the pin**: ragas 0.4 requires
  `langchain-openai<1`, which is why this project holds to API present in both
  langchain majors. A release that lifts that pin retires a standing
  constraint.
- [Hamel Husain's blog](https://hamel.dev/) — where evaluation methodology
  (error analysis, judge calibration) moves; `rag-experiment-methodology` is
  built on it.

Query phrases: *LLM judge calibration*, *faithfulness metric evaluation*, *RAG
evaluation benchmark contamination*.

## Always on: the dependency stack

- [LangChain changelog](https://changelog.langchain.com/) — this project moved
  to langchain 1.x on 2026-08-18 and the widget rides `create_agent`
  middleware; agent-API changes land here first.
- LangGraph releases — the agent extra's loop, `recursion_limit` semantics.
- [fastembed's supported models](https://qdrant.github.io/fastembed/examples/Supported_Models/)
  and the sentence-transformers Hub — the two embedding backends' catalogues.

## Always on: local models that could judge or answer

- [Ollama model library](https://ollama.com/library) — new local models worth
  screening. The standing rule: `.screens/` evidence has only ever been
  produced against `ollama`, so a new local model is a `raglab-judgescreen`
  run away from being allowed to grade — never allowed by reputation.

## How a finding from any source lands here

The sources feed four different sinks, and confusing them is how catalogues
rot:

1. **A new technique** → `rag-research-radar`'s triage, then a skill file.
2. **A new model** → the relevant catalogue (`EMBED_MODELS`, `OLLAMA_MODELS`,
   `CHAT_MODELS`) under the audit rule: the lists offer **only what has
   actually run here**, availability verified, licence in the label. A local
   judge additionally needs its screen run first.
3. **A new benchmark** → a candidate control corpus, through
   `datasets.validate()`'s contract — LongMemEval-style sessions would need
   the quote-verbatim rule satisfied by construction.
4. **A dependency release** → the pins note in `CLAUDE.md`, and only then the
   lockfile.

## In this lab

- The ordering above follows the bundled default corpus: it is
  memory-shaped and Farsi, so those two sections lead. Point
  `IndexConfig.dataset` at a different use case's corpus and the reading
  order changes with it — the evaluation and dependency sections are the
  constant.
- The memory section doubles as a fixture audit: LongMemEval's five abilities
  against `metrics.TYPES` — an ability the ground truth does not test is a gap
  in the fixture, not just in the pipeline.
- The radar's quarterly sweep should run the query phrases of whichever
  sections this installation's corpora make relevant, alongside its generic
  ones.

## Sources

- [LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory](https://arxiv.org/abs/2410.10813)
- [LongMemEval on GitHub](https://github.com/xiaowu0162/longmemeval)
- [LoCoMo: Conversational Memory Benchmark](https://www.emergentmind.com/topics/locomo)
- [FaMTEB: Massive Text Embedding Benchmark in Persian](https://arxiv.org/pdf/2502.11571)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [ragas releases](https://github.com/explodinggradients/ragas/releases)
- [LangChain changelog](https://changelog.langchain.com/)
