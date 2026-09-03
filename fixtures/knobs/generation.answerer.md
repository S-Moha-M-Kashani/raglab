# generation.answerer — what turns the retrieved context into an answer

- **Step:** Generation. **Default:** `extractive`.
- **Values:** `extractive`, `none`, `llm`.

## What the knob does
`none` measures retrieval alone. `extractive` quotes the longest sentence from
each of the top three chunks, tagged with its source document — deterministic,
free, and honest about quoting rather than answering. `llm` actually writes the
answer, in the corpus's own language, and must refuse when the corpus is silent.

## What it means scientifically
The three options are three points on the **extractive-to-abstractive** axis of
question answering, and they differ in what can go wrong:

- **`none`** is an *ablation*, and an important one. It isolates the retrieval
  stage so retrieval metrics can be read without generation variance in them. In
  experimental terms it removes a noisy, expensive component to make the
  measurement of the component under study cleaner.
- **`extractive`** is the classic span/sentence-selection reader. Its ceiling is
  low — it cannot synthesise, aggregate or paraphrase — but its **hallucination
  rate is structurally zero**: every character it emits came from the corpus,
  with the document id attached. That makes it the right control for asking how
  much of a generated answer's quality came from generation rather than from
  retrieval.
- **`llm`** is abstractive: it can fuse several chunks, resolve references and
  answer in the asker's register, and it introduces the two failure modes the
  judged metrics exist to catch — *unfaithfulness* (claims not supported by the
  context) and *incompleteness* (supported but missing the facts the question
  needed). Those are orthogonal: an answer can be perfectly faithful and nearly
  useless, which is exactly what this lab measured on the bundled diary
  (faithfulness 0.743 against fact coverage 0.261 — generation, not retrieval,
  was the bottleneck).

## Why RAG architectures have this knob
Because "RAG quality" is two stages multiplied together, and improving the wrong
one is the most common wasted effort in the field. Being able to fix generation
at a deterministic baseline — or remove it — is what makes retrieval claims
credible.

## When each option is useful
- **`none`** while tuning index and retrieval knobs: faster, free, no variance.
- **`extractive`** as the honest baseline and for auditing what retrieval
  actually delivered.
- **`llm`** for the real product behaviour, for cross-lingual answering, and
  whenever aggregation across chunks is required.

## Interactions
`generation.model` names the model for `llm`; `retrieval.grader` decides whether
refusal is even possible; `generation.fact_judge` and `run.ragas_mode` decide
how the answer is scored. Under the `fake` backend an `llm` answerer produces
invention, so a row's backend is part of its meaning.
