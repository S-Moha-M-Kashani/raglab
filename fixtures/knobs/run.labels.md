# run.labels — restrict the run to questions whose declared labels match

- **Step:** Run control. One switch-group per label the loaded ground truth
  declares with a closed set of values or a glossary — read from that dataset,
  so the choices differ from one corpus to the next.

## What the knob does
Filters the question set before sampling: only questions carrying the selected
label values are eligible.

## What it means scientifically
This is **subgroup analysis**, and it is both the most informative and the most
dangerous thing a run control can do.

Informative, because an aggregate metric hides mechanism. "Faithfulness 0.74" is
a mean over question kinds whose difficulty differs by construction — a lookup
question and a counting question exercise different parts of the pipeline. Slicing
by label is how you find *where* a configuration fails, which is a far more
actionable result than a single number moving.

Dangerous, for three well-known reasons:

- **Aggregation reversal (Simpson's paradox).** A configuration can win on every
  subgroup and lose overall, or the reverse, when subgroup sizes differ. So a
  filtered result is a statement about that subgroup, never about the corpus.
- **Multiple comparisons.** With enough slices, some subgroup will favour
  whichever arm you prefer. A filtered win discovered post hoc is a hypothesis,
  not a finding.
- **Comparability.** Two rows run on different label filters were measured on
  different question sets, so they are not comparable — which is precisely why
  this lab's ranking partitions by question set before naming any winner, and why
  the board treats the question set as a column rather than a detail.

Because the label vocabulary is read from the dataset rather than fixed by the
lab, the available slices are a property of the ground truth's authoring — a
corpus that declares no closed-set labels offers none.

## Why RAG architectures have this knob
Because different question types have different bottlenecks, and a pipeline
tuned on the average is often tuned for the most common type only. Filtering is
how you check the types you actually care about.

## When it is useful
- **Diagnosis**: run the same configuration on each difficulty band or question
  type to find where it breaks.
- **Targeted iteration** when a class of question matters more (unanswerable
  questions when testing a gate; counting questions when testing `card`
  summaries).
- **Never for headline numbers** without saying which subset produced them.

## Interactions
Restricts the pool `run.limit` and `run.balance` sample from; recorded on the row
because it defines the question set, which is part of a comparability group.
