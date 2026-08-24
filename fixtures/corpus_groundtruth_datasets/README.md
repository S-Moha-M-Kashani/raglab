# Bundled corpus/ground-truth pairs

This folder is where the lab keeps what it measures against, and it is the
place to start if you want to bring your own corpus. Read this file, look at
the templates, and copy one.

## What a pair is

A dataset is exactly two JSON files, joined by an id each one declares:

- `<name>_corpus.json` — what gets measured against: `corpus_dataset_metadata`
  (the id, a name, a language) and `corpus_documents` (an ordered list of
  documents, each an ordered list of text parts, plus whatever metadata the
  dataset chose to record and declared in `label_fields`).
- `<name>_groundtruth.json` — what a correct system should do:
  `groundtruth_dataset_metadata.corpus_ref.dataset` names the corpus above,
  and `groundtruth_dataset` is the list of questions, each with an
  `expected_answer` (what to say) and `relevant_corpus_documents` (what to
  retrieve, quoted as `evidence`).

There is no third shape. `diary-fa`, the bundled default, is an ordinary pair
like the other four — no native schema, no special-cased loader. Every pair
is checked, in full, against `schema_corpus.json` and `schema_groundtruth.json`
in this same folder — the machine-checked contract, run through the
`jsonschema` library plus the handful of cross-file rules a JSON Schema
cannot express (the pair actually joins, a quote is findable, a claim has
evidence). A file that fails is refused whole, every problem reported at
once — never repaired, never partially imported.

## Where the templates are

`corpus_template.json` and `groundtruth_template.json`, in this folder, are
the templates: a copy of every key the schema allows, each value replaced by
`requirement | type | what goes here`. Copy one, replace every value, delete
what you do not use, and validate the result against the schema next to it.
A convention test (`test_corpus_template_mirrors_corpus_schema_key_for_key`
et al. in `src/raglab/tests/test_conventions.py`) pins the two in step —
a key added to one and not the other fails there before it reaches an author
copying the template by hand.

The schema files themselves are not a template to fill in — they are the
JSON Schema `validate()` runs, with two extra blocks worth reading before you
write anything: `x-authoring` (the smallest valid shape, and a numbered
authoring checklist) and `x-acceptance` (below).

## The acceptance tiers

Both schemas declare an `x-acceptance` block: thresholds above the schema's
own floor, and what each one buys. The ground truth's is the one that
decides whether a run means anything, because every judged metric is a mean
over questions:

| tier | questions | buys you |
| --- | --- | --- |
| `well_formed` | 1 | Enough to validate the shape, and enough for a smoke run. |
| `measurable` | 8 | Below roughly this many, one bad question moves the mean more than the knob under test does. |
| `comparable` | 30 | The spread across the four judged metrics is narrower than the gap between two candidates — what lets a winner be named at all. |
| `decisive` | 100 | This project's own bar for keeping an experiment. Also roughly where per-question-label breakdowns stop being anecdotes. |

The corpus schema declares the same three lower tiers on **document** count
instead — `measurable` at 8 is where retrieval's default `k=8` stops handing
the answerer the whole corpus by accident, and `comparable` at 30 is where
moving one knob changes the retrieved set rather than reshuffling it. A
corpus this small (`smoke-mini`: 5 documents, 6 questions) is not meant to
measure anything — it is what you run end to end in seconds to check a
pipeline works before spending an hour on one that does not.

## The smallest valid pair

This validates against both schemas, `validate()` included, with zero
problems reported:

`tiny-example_corpus.json`:

```json
{
  "corpus_dataset_metadata": {
    "dataset": "tiny-example",
    "name": "Tiny example",
    "language": "en"
  },
  "corpus_documents": [
    {
      "corpus_document_id": 1,
      "document_content": [
        {"text": "The office moved to Zurich in March."}
      ]
    }
  ]
}
```

`tiny-example_groundtruth.json`:

```json
{
  "groundtruth_dataset_metadata": {
    "name": "Tiny example questions",
    "corpus_ref": {"dataset": "tiny-example"}
  },
  "groundtruth_dataset": [
    {
      "groundtruth_question_id": 1,
      "question": "Does the corpus mention a move to Paris?",
      "expected_answer": {"behavior": "abstain"},
      "relevant_corpus_documents": []
    }
  ]
}
```

One document, one question, no labels, no evidence — `behavior: "abstain"`
needs neither `expected_answer.text` nor a non-empty
`relevant_corpus_documents`, because nothing in the corpus answers it. It is
`well_formed` on both `x-acceptance` scales and nothing more: enough to
prove the shape, not enough to measure anything. `schema_corpus.json`'s own
`x-authoring.smallest_valid_corpus` and `schema_groundtruth.json`'s
`x-authoring.smallest_valid_question_set` show the same floor for a corpus
that *does* answer — `behavior: "answer"` with one verbatim quote — which is
the shorter read if abstention is not what you are trying to see.

## The six bundled pairs

Six corpora the lab ships with, one of them the bundled default (the English
diary). They are here to answer one question a single fixture could not: **is
this finding about retrieval, or about one corpus?** Each also stands in
for a use case from the map in `skills/rag-use-case-architectures/SKILL.md`,
so a row's suggested starting architecture can be tried against a corpus of
its own shape without leaving the repository.

| pair | language | documents | questions | what it is for |
| --- | --- | --- | --- | --- |
| `diary_year_en_corpus.json` / `diary_year_en_groundtruth.json` | English | 167 | 112 | The bundled default: a year of synthetic colloquial diary chat, the English rendering of the Farsi original below — same sessions, same questions. |
| `diary_year_fa_corpus.json` / `diary_year_fa_groundtruth.json` | Farsi | 167 | 112 | The Farsi original — the flagship case study, not the project's scope. An empty `dataset` still resolves to it, so recorded fingerprints keep their meaning. |
| `support_en_corpus.json` / `support_en_groundtruth.json` | English | 20 | 15 | A year of support conversations for a fictional analytics product. Latin script, short factual turns, dates and identifiers — the corpus where an ASCII embedder is genuinely competitive rather than scoring at chance, which is the control the Farsi finding never had. |
| `meetings_de_corpus.json` / `meetings_de_groundtruth.json` | German | 15 | 12 | Weekly meeting notes for a fictional software team. A third script family and compound nouns, where an English-only embedder separates sharply from a multilingual one. Each question carries an English translation. |
| `research_multihop_corpus.json` / `research_multihop_groundtruth.json` | English | 18 | 14 | Reading notes following one replication dispute. Deliberately weighted to multi-hop and aggregation, declared as question labels: most answers need two or three documents, so a pipeline that retrieves one perfect chunk still scores badly. |
| `smoke_mini_corpus.json` / `smoke_mini_groundtruth.json` | English | 5 | 6 | Too small to measure anything and not meant to — the fastest end-to-end check that a pipeline works before spending an hour on one that does not. Declares no date label and no `ranks` label on purpose, so the suite's fastest corpus also exercises those absent-declaration paths on every run. |

All six are synthetic. Every person, company, product and finding in them
is fictional; none of it is anybody's data.

They are read-only reference points. Anything imported through the panel
lands in `.datasets/` instead, which is git-ignored.

## What makes them trustworthy

Every one satisfies the contract `dataset_import_contract.validate()`
enforces, and `src/raglab/corpora/tests/test_datasets.py` re-checks that on
every run of the suite. The rule that matters most: **each evidence quote
whose `fidelity` is `"verbatim"` appears character for character in the
document it cites.** Quote recall, the Inspector's highlighted evidence
spans and the offline RAGAS context metrics are all computed against those
strings, so a ground truth that misquotes its corpus does not score worse —
it scores confidently about text that was never there.

They were authored so that the rule holds by construction rather than by
proofreading: each quote is a variable in the authoring script and the
document it belongs to is written by interpolating that same variable, so
the two cannot drift apart.

## What they are not

They are small. Fifteen questions cannot decide an architecture — the
deciding metrics are means over questions, and a mean over fifteen has an
error bar wide enough to swallow most of the candidate field. Use them the
way a control is used: to find out whether something measured on the diary
survives contact with a different language, a different domain, or a
different question shape.
