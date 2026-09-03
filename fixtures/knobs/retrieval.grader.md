# retrieval.grader — the gate that makes abstention possible

- **Step:** Retrieval. **Default:** `none`.
- **Values:** `none`, `lexical`, `llm`.

## What the knob does
Chunks scoring below `retrieval.grade_threshold` are dropped, and if nothing
survives, the pipeline **refuses** instead of answering from noise. With `none`,
every question gets an answer — including the ones the corpus never mentions.

## What it means scientifically
This is **selective prediction** (also: classification with a reject option)
applied to retrieval. The system is allowed to output "I cannot answer", which
turns the evaluation from pure accuracy into a trade-off between two error
types:

- **False refusal:** the corpus does contain the answer and the gate discarded
  it — a recall failure the user experiences as unhelpfulness.
- **False answer:** the corpus is silent and the pipeline answers anyway — the
  failure mode that produces confident fabrication, because a generator handed
  irrelevant context will still write something.

Retrieval-quality gating is the mechanism behind the self-reflective RAG
architectures (Self-RAG's critique tokens, CRAG's retrieval evaluator): judge
the retrieved evidence *before* generating, and change behaviour when it is
weak. It matters because generators do not reliably abstain on their own —
irrelevant context degrades answers rather than producing refusals.

The choice between `lexical` and `llm` is a choice of classifier. A lexical
gate scores term overlap: cheap, deterministic, and blind to paraphrase — so its
score distribution for relevant and irrelevant chunks overlaps heavily. An LLM
gate reads the pair and can recognise topical relevance, at the cost of a call
per candidate and whatever calibration the model happens to have.

## Why RAG architectures have this knob
Because a benchmark of answerable questions hides the whole problem. A corpus is
finite; some questions are outside it; and a pipeline with no gate scores exactly
the same whether it knows that or not. The gate is what makes "unanswerable" a
measurable outcome — which is why the bundled ground truths declare an `abstain`
behaviour at all.

## When it is useful
- **Any deployed system** where a wrong answer costs more than no answer.
- **Whenever the question set contains unanswerable questions** — otherwise the
  gate can only hurt, and measuring it is measuring only its false refusals.
- **`llm` over `lexical`** when both false-error rates must be low at once:
  measured on the bundled diary, the lexical gate had no usable operating point
  while an LLM gate at 0.4 did.

## Interactions
`retrieval.grade_threshold` is the operating point; `retrieval.grader_model`
names the model for `llm`. A refusal changes what the generation metrics mean, so
gate settings belong on the row.
