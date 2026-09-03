# generation.fact_judge — score the answer against the ground truth's atomic facts

- **Step:** Generation. **Default:** off.
- **Reported, non-voting:** fact coverage does not decide a configuration; the
  four judged metrics do.

## What the knob does
Scores each answer against the ground truth's atomic `derived_facts` with a
model: for each declared fact, is it present in the answer?

## What it means scientifically
This is **decomposed / nugget-based evaluation**. Instead of comparing an answer
to a reference string, the reference is broken into atomic claims and each is
checked independently — the design behind TREC's nugget-based assessment and,
more recently, FActScore-style atomic factuality scoring. Three reasons it is the
better instrument for completeness:

1. **String similarity measures the wrong thing.** Two answers containing the
   same facts in different words score differently under overlap metrics; two
   answers with different facts and similar wording score alike. Atomic facts
   remove the phrasing from the comparison.
2. **It measures coverage, not faithfulness.** Faithfulness asks "is everything
   in the answer supported by the context"; coverage asks "is everything the
   question needed in the answer". They are orthogonal, and a system can be
   excellent at one and poor at the other — measured here, faithfulness 0.743
   against coverage 0.261, which is what exposed generation as the bundled
   corpus's bottleneck.
3. **It is the only option across languages.** When the answer and the reference
   are not in the same language — the bundled diary answers in Farsi against
   English facts — no lexical metric can compare them at all. The judge is
   translating as well as judging, which is why a weak model here produces
   confidently wrong scores rather than noisy ones.

## Why RAG architectures have this knob
Because "did it answer the question" is not a single quantity, and the
completeness half is invisible to the metrics that check grounding. Making fact
coverage explicit — and reported rather than voting — keeps the deciding metrics
stable while still surfacing the failure they miss.

## When it is useful
- **Turn it on** when answers look good and users still say they are unhelpful:
  that is the coverage failure this catches.
- **Essential for cross-lingual setups**, where it is the only working
  completeness metric.
- **Requires a ground truth with `derived_facts`** declared; without them there
  is nothing to check.

## Interactions
`generation.judge_model` names the model. Reported alongside the four deciding
metrics but never averaged into them; `run.ragas_mode = judged` is what produces
those four.
