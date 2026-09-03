# run.dataset-file — importing a corpus and its ground truth

- **Step:** Run control (the file input beside `index.dataset`).
- **A dataset is two JSON files, paired by id:** a corpus, and the ground truth
  measured against it. Both are checked in full against the schemas in
  `fixtures/corpus_groundtruth_datasets/` and **refused — never repaired** —
  with every problem reported at once.

## What the knob does
Loads your own corpus into the lab. Start from `corpus_template.json` and
`groundtruth_template.json`, replace every example value, delete what you do not
use. Imports land in `.datasets/`; the dataset id — not a filename — is what
every run and leaderboard row records.

## What the contract asks for, and why
**The corpus:** documents made of parts, plus a declaration of every label used
anywhere (`label_fields`), stating its type, the levels it may apply to, and —
for a closed set — its allowed values. Two label types are wired to pipeline
behaviour: one typed `date-time` drives time filtering and recency; one numeric
label declaring `ranks: true` drives importance. Neither is required, and their
absence is what greys out `retrieval.time_filter`,
`retrieval.recency_half_life_days` and `retrieval.agentic_weights`.

**The ground truth:** for each question, an `expected_answer` with a `behavior`
— `answer`, `abstain` (nothing in the corpus answers it) or `correct_premise`
(a false premise, corrected from the corpus) — a reference `text`, the atomic
`derived_facts` a correct answer must contain, and the relevant documents with
`evidence` quotes marked `verbatim`, `paraphrase` or `computed`.

## What it means scientifically
This is **instrument validity**, and it is the one part of a RAG evaluation that
no downstream sophistication can repair.

- **The measured object is the pair, not the corpus.** Retrieval metrics are
  defined against declared relevance; a corpus without a ground truth is not
  measurable, and a ground truth that disagrees with its corpus produces
  confident numbers about text that does not exist.
- **The verbatim rule is the load-bearing one.** Every `verbatim` evidence quote
  must appear character for character in the document it cites, because quote
  recall, the Inspector's highlighted spans and the offline context metrics are
  all computed against those strings. A ground truth that misquotes its corpus
  does not score *worse* — it scores confidently about text it never contained.
  That is the difference between a noisy instrument and a broken one.
- **Declaring `abstain` questions is what makes abstention measurable.** Without
  them, a pipeline that answers everything and a pipeline that knows its limits
  score identically, and `retrieval.grader` can only appear to hurt.
- **Fidelity labels keep metrics honest.** Only `verbatim` quotes may be scored
  lexically; `paraphrase` and `computed` evidence exists in the corpus's meaning
  but not its characters, and pretending otherwise would penalise correct
  retrieval.
- **Refusing rather than repairing** is the right failure mode for an instrument:
  a silently corrected dataset is one whose recorded identity no longer matches
  what was measured.

## When it is useful
Whenever the question is whether a finding on the bundled corpora generalises to
*your* material — which is the only question the lab's control corpora exist to
ask.

## Interactions
Sets `index.dataset` (the id, and therefore the index fingerprint and the
leaderboard's grouping); its declared labels enable or grey out four retrieval
knobs and populate `run.labels` and `run.balance`; its `derived_facts` are what
`generation.fact_judge` scores against.
