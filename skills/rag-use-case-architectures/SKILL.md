---
name: rag-use-case-architectures
description: Choose a starting RAG architecture from the shape of the use case — the corpus, the question distribution, the update rate, the cost of a wrong answer — instead of from whatever technique is newest. Use at the start of any new RAG project, when porting a pipeline to a different domain, or when someone asks "which RAG approach for X". Maps ten recurring use cases (support, legal, finance, medical, code, sensemaking, multi-hop research, personal memory, freshness-critical, unified search) to measured starting points, with the case-study evidence behind each and the first thing to measure before believing the choice.
---

# RAG Architectures by Use Case

**What it is.** An architecture is not chosen from the technique literature; it
is chosen from the use case, and the technique literature says which lever fits
which shape of problem. The industry case studies of 2026 agree on where RAG
pays at all: knowledge-intensive, regulated domains — finance, legal, healthcare
— where document volumes are large, wrong answers are expensive, and **source
attribution is required**, which is the property retrieval has and a bare model
does not.

## The five questions that determine the architecture

Ask these before naming any technique:

1. **What do the questions look like?** Factoid lookups, aggregates over the
   whole corpus, multi-hop chains, comparisons, conversational follow-ups. Get
   the *distribution*, not the most interesting example — the architecture
   serves the median question and must not fail the tail.
2. **What does the corpus look like?** Size, structure (markup? tables? code?),
   language, whether units are self-contained or anaphoric, how often it
   changes.
3. **What does a wrong answer cost?** This sets the abstention design and the
   evaluation budget. A support bot that guesses is annoying; a clinical or
   legal one that guesses is a liability. High cost → gates, citations,
   abstention, judged evaluation.
4. **What is the latency and cost budget?** Per-query LLM calls (agents, HyDE,
   LLM rerankers) versus index-time costs (contextual retrieval, hierarchies)
   versus one-time costs (encoder choice).
5. **Who checks the answer?** A cited answer a human verifies tolerates lower
   precision than an answer consumed by another program.

## The map

| Use case | Corpus shape | Question shape | Starting architecture | Measure first |
| --- | --- | --- | --- | --- |
| **Customer support / FAQ** | tickets, KB articles; short, self-contained | repetitive factoids + follow-ups | light or no chunking, hybrid retrieval, conversational rewriting, strict grounding with abstention | deflection vs escalation; false-answer rate |
| **Enterprise unified search** | wikis, docs, chats, tickets — heterogeneous | "where is / what is our policy on" | hybrid + reranker; per-source metadata filters; cited answers | recall@20 per source type |
| **Legal / contracts / regulatory** | long, structured, cross-referenced, anaphoric | clause lookup, precedent, obligations | structure-aware chunking, contextual retrieval, cross-encoder rerank, verbatim citations | citation exactness; retrieval failure on defined terms |
| **Financial filings / reporting** | entity-dense, numeric, tabular | "X's revenue in Q2", KYC, audit trails | contextual retrieval (its original case study), hybrid — BM25 carries tickers and figures — table-aware parsing | top-20 failure rate; numeric fidelity |
| **Medical / clinical support** | literature + institutional protocols | evidence-backed recommendations | corrective gate before generation, judged faithfulness bar, abstention as first-class outcome | faithfulness; unsupported-claim rate |
| **Code / API documentation** | code + docs; identifiers everywhere | "what calls X", "how do I use Y" | AST/structure chunking, lexical-heavy hybrid (identifiers are BM25's home game), symbol-aware metadata | exact-identifier recall |
| **Global sensemaking** | one large corpus, themes wanted | "main themes", counts, trends | hierarchical: RAPTOR or community summaries; flat retrieval cannot answer these at all | whether summary rows are retrieved for aggregate questions |
| **Multi-hop research** | facts scattered across documents | chains, comparisons, syntheses | agentic retrieval loop with caps, or decomposition; graph structure if entities recur | evidence sufficiency per hop; stop reasons |
| **Personal / agent memory** | dialogue sessions, timestamped, one voice | "when did I", "how often", time-scoped | session/turn chunking, recency + time filters, language-matched encoder, hierarchy for the counting tail | recall by question type; time-filter precision |
| **Freshness-critical (news, prices)** | high update rate | about the latest state | incremental indexing (LightRAG-style), recency ranking, no expensive index-time enrichment you must repay per update | staleness of retrieved evidence |

Two 2026 patterns sit above single rows: **adaptive routing** — classify the
query and select retrieval depth per query, so simple operational questions do
not pay the deep pipeline (see `adaptive-corrective-rag`) — and **federated
retrieval** for cross-organisational cases (hospitals, jurisdictions, banks)
where the corpus cannot be centralised and retrieval goes to the data rather
than the reverse.

## How to read the map honestly

- Each row is a **starting point** — the educated guess that becomes candidate
  zero — not a conclusion. The case studies behind these rows are other
  people's corpora; a row earns nothing on yours until measured there. See
  `rag-experiment-methodology` for what happens next.
- Rows compound: a legal assistant with follow-up questions is the legal row
  plus conversational rewriting; a Farsi support bot is the support row plus
  everything in `multilingual-rag`.
- The most common error is over-architecting the median: if 80% of questions
  are factoid lookups, the hierarchy and the agent serve the other 20% and must
  be measured on that 20% (`rag-evaluation`'s per-type rule), while the 80% pay
  their latency.

## In this lab

- The lab **is** the personal-memory row, built out: `session`/`turn-pair`/
  `message` chunkers, `recency` reranker with `recency_half_life_days`, the
  Farsi `time_filter`, a language-matched encoder, and `hierarchy` for the
  counting tail — that row's whole recipe exists as knobs.
- The five questions above are answerable here with unusual precision because
  the ground truth carries `types` and `difficulty` per question: the question
  distribution is a query away, not a guess.
- The four control corpora map onto other rows — support tickets (support row),
  meeting notes (sensemaking-adjacent), research notes weighted to multi-hop
  (research row) — so a row's recipe can be tested against a corpus of its own
  shape without leaving the lab.
- What the lab cannot express from this table: table-aware parsing, federated
  retrieval, incremental indexing (an index is rebuilt, never updated — a
  deliberate property of the in-memory store).

## Sources

- [Top 10 enterprise use cases for RAG models in 2026 — Glean](https://www.glean.com/perspectives/top-10-enterprise-use-cases-for-rag-models-in-2026)
- [10 RAG Architectures in 2026: Enterprise Use Cases & Strategy — Techment](https://www.techment.com/blogs/rag-architectures-enterprise-use-cases-2026/)
- [Enterprise RAG Guide 2026: Modular, GraphRAG & Agentic Patterns](https://www.synvestable.com/enterprise-rag.html)
- [How to Build an Enterprise RAG System in 2026 — Vention](https://ventionteams.com/blog/enterprise-rag-implementation-guide)
- [RAG Examples: 15 Real Use Cases from Companies in 2026](https://www.startdesigns.com/blog/rag-examples/)
- [Contextual Retrieval — Anthropic Engineering](https://www.anthropic.com/engineering/contextual-retrieval) (the financial-filings case study)
