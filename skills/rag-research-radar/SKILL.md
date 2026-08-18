---
name: rag-research-radar
description: Keep this skills corpus current — where new RAG research is announced, how to run a periodic sweep for techniques worth adopting, how to triage a paper's claim before believing it, and the procedure for folding a new technique into skills/ as a properly formatted SKILL.md with an "In this lab" mapping. Use on a recurring cadence (quarterly is enough), when someone names a technique not covered here, or when a skill's frontier snapshot has gone stale. Carries a dated snapshot of the 2026 frontier so staleness is visible rather than silent.
---

# RAG Research Radar

**What it is.** A prose corpus about a fast-moving field rots by default. This
skill is the maintenance procedure: where to look, what to believe, and how a
finding becomes a file here. Its own frontier snapshot is dated so that its
staleness is a fact on the page, not a discovery.

## Where new research is announced

**Primary — where papers land first:**

- [arXiv cs.CL](https://arxiv.org/list/cs.CL/recent) — computation and language; most RAG papers post here
- [arXiv cs.IR](https://arxiv.org/list/cs.IR/recent) — information retrieval; fusion, reranking, late interaction
- [arXiv cs.AI](https://arxiv.org/list/cs.AI/recent) — agentic and memory systems
- [Hugging Face Daily Papers](https://huggingface.co/papers) — community-curated daily cut; the highest signal-per-minute skim
- [OpenReview](https://openreview.net/) — submissions with their reviews visible, which is triage done for you
- [ACL Anthology](https://aclanthology.org/) — the *CL conferences' proceedings, post peer review
- [Semantic Scholar](https://www.semanticscholar.org/) — citation alerts on seed papers (RAPTOR, Self-RAG, ColBERT) surface successors automatically

**Venues whose proceedings are worth a pass per cycle:** SIGIR, ECIR (its LIR
workshop is where late-interaction work concentrates), EMNLP, ACL, NAACL,
NeurIPS, ICLR.

**Aggregators and applied sources — where measured practice lands:**

- [The Turing Post](https://www.turingpost.com/) — periodic RAG-type roundups
- [awesome-generative-ai-guide's RAG research table](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/research_updates/rag_research_table.md) — maintained paper index
- Engineering blogs that publish numbers, not announcements: [Anthropic Engineering](https://www.anthropic.com/engineering) (contextual retrieval came from here), Jina, Meilisearch, Qdrant/Weaviate — vendor-biased toward their own store, which is fine once you know it
- [Hamel Husain's blog](https://hamel.dev/) and the eval-practice circle around it — where evaluation methodology (not architecture) moves

## Triage: what to believe

A technique earns a skill here by clearing four questions, in order:

1. **Is the claim measured, and against what?** A named benchmark with a named
   baseline, or prose. The baseline matters more than the number — beating
   naive RAG is 2023's bar; the honest 2026 baselines are hybrid + rerank and
   a well-tuned single pass.
2. **Does it have a control?** A graph method that never ran against k-means
   clustering, an agent that never ran against a widened top_k — the win may
   belong to the cheap half. Papers that publish their ablations are the
   minority worth reading closely.
3. **Where does the cost land?** Index time (paid per rebuild), query time
   (paid per question forever), or training time (paid once, needs data). A
   method's adoptability here is mostly this answer.
4. **Does it survive this corpus's constraints?** Farsi (does the method assume
   an English extractor?), no model at build time, an in-memory store with one
   vector per row. Many strong papers fail one of these, which makes them a
   stated absence, not a candidate.

## The update procedure

Quarterly, or on demand:

1. Sweep the primary sources with queries shaped like: *advanced RAG techniques
   {year}*, *{seed paper} successor*, *RAG survey {year}*, plus one per
   existing skill's topic. Follow citation alerts on the seed papers.
2. Triage each candidate technique with the four questions above.
3. For a technique that clears them: new folder under `skills/`, `SKILL.md`
   with `name` matching the folder and a description stating what it does *and
   when to use it* (the format rules live in `skills/README.md`), literature
   numbers labelled as that corpus's result, and a closing **"In this lab"**
   section that maps it to existing knobs or names precisely what is missing —
   a knob, a store change, a rule it would break. That last section is the
   admission price: a skill that cannot say how it meets this codebase is a
   blog post.
4. For a technique that fails triage but keeps appearing: one line in the
   frontier snapshot below, so the next sweep starts warm.
5. Re-date the snapshot. Update any older skill whose "In this lab" section a
   new lab feature has made stale — the skills describe the lab, and the lab
   moves.

## Frontier snapshot — 2026-08-18

Sighted and triaged, not adopted. Grouped by what they attack:

- **Memory-structured retrieval** — the field's current center of gravity, and
  directly relevant to a diary-memory lab: **HGMem** (hypergraph working
  memory; hyperedges as memory units for multi-step reasoning), **xMemory**
  (agent memory retrieval by decoupling and aggregation), **MemAdapter**
  (generative subgraph retrieval across memory paradigms), **MiA-RAG** and
  **Disco-RAG** (global context and discourse structure over long documents),
  **SMMBench** (benchmark for source-distributed multimodal agent memory).
- **Multi-hop evidence organisation** — **HKVM-RAG** (key-value-separated
  hypergraph evidence), **Graph-O1**, **ConRAG** (consensus multi-view
  retrieval), **GraphSearch** (agentic deep search over graph RAG).
- **Deciding when/what to retrieve** — **Bidirectional RAG**, **A-RAG**,
  **SURE-RAG**, **QuCo-RAG**, predictive prefetching; the adaptive-routing
  benchmarks (`adaptive-corrective-rag` carries the numbers).
- **Cheap graph construction** — **LinearRAG**, **LiteSemRAG**, token
  co-occurrence graphs, **TagRAG**: the race to keep GraphRAG's accuracy and
  delete its LLM calls (`hierarchical-graph-rag` covers the established end).
- **Multimodal RAG** — retrieval across text, tables, images, video
  ([survey repo](https://github.com/llm-lab-org/multimodal-rag-survey));
  **FLOWREADER** (min-cost-flow over multimodal long documents). Out of scope
  for a text diary; the likeliest to matter for other corpora.

Common thread worth quoting: retrieval research has moved from *finding
similar chunks* to *organising evidence* — graphs, hypergraphs, memory
structures, discourse — which is the same direction this lab's hierarchy knob
already points.

## In this lab

- The four triage questions are this repository's own rules generalised: the
  control requirement is `metadata`/`kmeans`'s role, the cost question is the
  fingerprint-versus-retrieval-knob split, the constraint check is the
  no-model-at-build rule and the Farsi zero-vector test.
- The memory-structured group above is the one to watch closely: this lab *is*
  an agent-memory retrieval bench, and HGMem-style structures are the first
  frontier family aimed at its exact problem rather than at Wikipedia QA.
- A sweep of this radar is manual today. If the widget grows a research tool
  later, this file's source list and triage rubric are its spec.

## Sources

- [20 Advanced RAG Types to Know in 2026 — Turing Post](https://www.turingpost.com/p/ragtypes)
- [HGMEM: Hypergraph-based Working Memory for Multi-step RAG](https://arxiv.org/abs/2512.23959)
- [xMemory: Beyond RAG for Agent Memory](https://github.com/HU-xiaobai/xMemory)
- [HKVM-RAG: Hypergraph Evidence Organization for Multi-Hop RAG](https://arxiv.org/pdf/2606.07218)
- [Multimodal RAG Survey](https://github.com/llm-lab-org/multimodal-rag-survey)
- [Most Impactful RAG Papers — maintained index](https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/research_updates/rag_research_table.md)
