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

## The experiment ladder, per use case

The map gives candidate zero; this section gives the order of the experiments
that follow — first try, then check, then add. Every rung obeys
`rag-experiment-methodology`: one knob per candidate, error bars, and the next
rung chosen by the dominant failure class, not by the ladder alone. The
orderings encode where each use case's yield usually is, so they are educated
guesses to be corrected by measurement, not scripts.

**For the use case of personal / agent memory** (diary, assistant memory):
first try session or turn chunking, a language-matched encoder verified on the
script, dense retrieval, extractive answers. Then check recall per question
type — the aggregate/counting tail fails differently from factoids. Then add
hybrid-RRF (names, dates and numbers are literal tokens) and a time treatment
— a time *filter* for "when did I", *recency* ranking for "current state" —
measuring each alone. Then check recall@rerank_depth against recall@k before
paying for any reranker; add one only if the gap is large. Then add a
relevance gate if fabricated memories are the risk that matters. Only then a
hierarchy, measured on the counting tail and never on the mean; only then an
agent scope, against a widened top_k control.

**For the use case of team meeting notes / organisational knowledge**: first
try structure-aware chunks (one per agenda item or decision, speaker tags
kept), hybrid retrieval from the start — project codes and names make BM25
non-optional — and metadata filters on date, attendees, project. Then check
follow-up questions separately from first questions; if follow-ups fail, add
conversational rewriting before touching retrieval. Then add the
corpus-declared (metadata) hierarchy — "all decisions about X" is a mainstream
question here — and check it on exactly those questions. Then add recency
ranking for "current status" questions. Keep index-time enrichment cheap: the
corpus updates weekly, and every rebuild repays it.

**For the use case of customer support / FAQ**: first try minimal or no
chunking (units are already question-shaped), hybrid retrieval, strict
grounding with abstention. Then check the false-answer rate before anything
else — a wrong confident answer costs more than an escalation. Then add
conversational rewriting (support is dialogue), then a reranker only if
recall@20 is fine while top-3 is wrong.

**For the use case of legal / regulated documents**: first try structure-aware
chunking on the document's own units (clauses, sections), hybrid retrieval,
and verbatim citation in the answer path. Then check retrieval on defined
terms and cross-references — the characteristic failure. Then add a
cross-encoder reranker (precision is the product here), then contextual
enrichment of chunks — static header first, LLM blurb only if the static
header measurably falls short, because the corpus is anaphoric enough to be
the technique's best case.

**For the use case of entity-dense financial text**: first try hybrid with the
lexical arm weighted seriously (tickers, figures), table-aware parsing if
tables carry the answers, and a static situating header (company, period) on
every chunk. Then check top-20 failure rate — this is contextual retrieval's
home corpus, and the header-vs-LLM-blurb comparison is worth running here if
anywhere. Then add numeric-fidelity checks to evaluation before trusting any
score.

**For the use case of global sensemaking** (themes, trends, "what is this
corpus about"): first try a hierarchy — flat retrieval cannot answer these at
all, so the baseline is honest about failing them. Then check whether summary
rows are actually retrieved for aggregate questions (`n_summaries` is the
model: count it, per row). Then compare groupings against the two controls —
the corpus's declared structure and naive clustering — before believing any
graph method.

**For the use case of multi-hop research questions**: first try query
decomposition over a strong single-pass baseline. Then check where hops fail —
evidence missing versus evidence unreachable by the first query's vocabulary.
Then add an agentic retrieval loop with caps and stop reasons, always beside a
widened-top_k control, because the loop must beat the cheap alternative before
its cost is justified.

Automating this ladder — a runner that walks candidates and records rows — is
what a sweep is. **The sweep's interface here is deliberately not finalized**:
it is planned as a widget tool, and until then the ladder is run by hand, one
candidate at a time, which the methodology requires anyway.

## In this lab

- The lab is generic across these rows — the corpus is a config field — and
  the bundled default corpus makes it the personal-memory row out of the box:
  `session`/`turn-pair`/`message` chunkers, `recency` reranker with
  `recency_half_life_days`, the time filter, a language-matched encoder, and
  `hierarchy` for the counting tail. That row's recipe exists as knobs; the
  other rows' recipes are the same knobs pointed at their corpora.
- A per-use-case preset dropdown on the panel is planned, not built — the
  production preset button (`#use-production`, fills and never runs) is the
  pattern it would generalise, one preset per row of the map above.
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
