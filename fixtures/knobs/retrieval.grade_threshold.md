# retrieval.grade_threshold — the score a chunk must clear to survive the gate

- **Step:** Retrieval. **Default:** 0.0 (nothing is dropped).

## What the knob does
Sets the cutoff for `retrieval.grader`. Chunks below it are discarded; if none
survive, the pipeline abstains.

## What it means scientifically
This is the **operating point** on the gate's ROC curve, and the only honest way
to talk about it is with both error rates at once:

- Raising the threshold catches more unanswerable questions (fewer false
  answers) and wrongly refuses more answerable ones (more false refusals).
- The question is never "what is the best threshold" but "does this classifier
  have *any* threshold where both error rates are acceptable" — that is, is
  there separation between the score distributions of relevant and irrelevant
  chunks at all.

Measured on the bundled diary, that distinction decided the choice of gate:

- The **lexical** gate had no usable setting. At 0.6 it caught 6 of the 8
  unanswerable questions but wrongly refused 52% of the answerable ones — the
  distributions overlap, so no cutoff separates them.
- The **LLM** gate at 0.4 refused all 5 unanswerable questions with a 3% false
  refusal rate — separation good enough for a working operating point.

That pair of results is the argument for the knob's existence: a threshold is a
property of the *classifier*, not of the corpus, and a scale that cannot
separate cannot be fixed by tuning.

## Why RAG architectures have this knob
Because abstention is a policy decision with an asymmetric cost, and the cost
asymmetry differs by application. A medical or legal assistant should refuse
readily; an exploratory research tool should not. One number expresses that
policy.

## When it is useful
- **Tune it against a question set containing known-unanswerable questions**;
  without them, only the false-refusal half is observable and every increase
  looks harmful.
- **Report both rates** whenever quoting a threshold — a threshold without its
  two error rates is not a result.
- **Leave it at 0.0** when `retrieval.grader = none`, where it does nothing.

## Interactions
Meaningful only with a grader; the scale differs between graders, so a threshold
carried over from `lexical` to `llm` means nothing. Interacts with
`retrieval.k` and `retrieval.candidates`, since the gate runs on what survived
them.
