---
name: rag-source-watchlist
description: The named sources to search for new RAG information relevant to this project specifically — one watchlist entry per lab concern (conversational-memory retrieval, Persian embeddings, evaluation practice, the LangChain/ragas dependency stack, local judge models), each with the links to check and the query phrases that find its literature. Use when looking for new work on this lab's own problems rather than the field at large, when a dependency or model catalogue needs updating, or when a benchmark for diary-style memory is wanted. The generic sweep procedure and field-wide venues live in rag-research-radar; this skill is where that procedure points for this repository.
---

# RAG Source Watchlist

**What it is.** `rag-research-radar` answers "where does RAG research land and
how is it triaged" — field-wide, corpus-agnostic. This skill is the other half:
**this project's** concerns, each mapped to the sources that carry news about
it and the query phrases that find that literature. The radar is the
procedure; this is its address book for `raglab`.

## Concern 1 — conversational-memory retrieval (the lab's own problem)

The diary fixture — timestamped multi-session chat plus questions requiring
recall across sessions — is a recognised benchmark shape with its own
literature, and it is the highest-value thing to watch, because findings there
transfer to this lab almost without translation.

- [LongMemEval](https://arxiv.org/abs/2410.10813) ([code](https://github.com/xiaowu0162/longmemeval),
  [site](https://xiaowu0162.github.io/long-mem-eval/)) — ICLR 2025; ~500
  questions over timestamped chat histories, testing **five abilities:
  information extraction, multi-session reasoning, temporal reasoning,
  knowledge updates, abstention**. That list is close to a taxonomy of this
  lab's own question types, and its finding — commercial assistants drop ~30%
  accuracy on sustained interaction — is the problem statement this lab works
  on.
- [LoCoMo](https://www.emergentmind.com/topics/locomo) — 50 conversations, up
  to 35 sessions, ~300 turns each; single-hop, multi-hop, temporal and
  open-domain questions over dialogue history. The nearest published analogue
  to `diary_year_fa.json`'s design.
- The memory-structured frontier tracked in `rag-research-radar`'s snapshot
  (HGMem, xMemory, MemAdapter) — agent-memory systems whose retrieval half is
  this lab's subject.

Query phrases: *conversational memory retrieval*, *multi-session temporal
reasoning QA*, *long-term memory chat assistant benchmark*, *lifelog / diary
question answering*, plus citation alerts on the two benchmarks above.

## Concern 2 — Persian and multilingual embeddings

The embedder is the lab's single most consequential component (the 60×
finding), so a better Persian encoder is the one drop-in upgrade always worth
checking for.

- [FaMTEB](https://arxiv.org/pdf/2502.11571) — the Persian MTEB; the benchmark
  that should decide a Persian encoder, not a general leaderboard position.
- [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) — filter
  by language; where new multilingual encoders surface first.
- [heydariAI/persian-embeddings](https://huggingface.co/heydariAI/persian-embeddings)
  — the current default's model page; its neighbours on the Hub are the
  candidate pool.
- [Advancing RAG for Persian (LREC 2026)](https://aclanthology.org/2026.lrec-1.580/)
  — the Persian-RAG best-practice line, including MatinaSRoberta.

Query phrases: *Persian text embedding benchmark*, *Farsi sentence
transformer*, *multilingual E5 successor*.

## Concern 3 — evaluation practice and the judge

- [ragas documentation](https://docs.ragas.io/) and
  [ragas releases](https://github.com/explodinggradients/ragas/releases) —
  watched for two reasons: metric changes (a changed judge prompt moves every
  judged number, silently) and **the pin**: ragas 0.4 requires
  `langchain-openai<1`, which is why the lab holds to API present in both
  langchain majors. A ragas release that lifts that pin retires a standing
  constraint.
- [Hamel Husain's blog](https://hamel.dev/) — where evaluation methodology
  (error analysis, judge calibration) moves; `rag-experiment-methodology` is
  built on it.

Query phrases: *LLM judge calibration*, *faithfulness metric evaluation*,
*RAG evaluation benchmark contamination*.

## Concern 4 — the dependency stack

- [LangChain changelog](https://changelog.langchain.com/) — the lab moved to
  langchain 1.x on 2026-08-18 and the widget rides `create_agent` middleware;
  agent-API changes land here first.
- LangGraph releases — the agent extra's loop, `recursion_limit` semantics.
- [fastembed's supported models](https://qdrant.github.io/fastembed/examples/Supported_Models/)
  and the sentence-transformers Hub — the two embedding backends' catalogues.

## Concern 5 — local models that could judge or answer

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

- This watchlist is scoped by the lab's own measured priorities: embedder
  first (60×), evaluation second (the judged metrics decide everything),
  memory-retrieval literature third (the problem itself), stack and models as
  maintenance. A generic RAG feed inverts that order.
- Concern 1 doubles as a roadmap source: LongMemEval's five abilities are a
  checklist against `metrics.TYPES` — an ability the ground truth does not
  test is a gap in the fixture, not just in the pipeline.
- The radar's quarterly sweep should run this file's queries alongside its
  generic ones; the two skills split what one oversized file would blur.

## Sources

- [LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory](https://arxiv.org/abs/2410.10813)
- [LongMemEval on GitHub](https://github.com/xiaowu0162/longmemeval)
- [LoCoMo: Conversational Memory Benchmark](https://www.emergentmind.com/topics/locomo)
- [FaMTEB: Massive Text Embedding Benchmark in Persian](https://arxiv.org/pdf/2502.11571)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [ragas releases](https://github.com/explodinggradients/ragas/releases)
- [LangChain changelog](https://changelog.langchain.com/)
