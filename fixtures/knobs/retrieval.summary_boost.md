# retrieval.summary_boost — multiply every summary's score before the cut

- **Step:** Retrieval. **Default:** 1.0 (off).
- **Applied before the candidate cut, never after.**

## What the knob does
Scales every summary row's retrieval score so summaries compete more strongly
against leaves.

## What it means scientifically
A boost is a **prior on a row type**, and this knob is a case study in two
things.

**Where it is applied matters more than its value.** A boost applied *after* the
candidate cut cannot promote anything: a summary that had not already survived
the cut is not in the list being reordered. That version was measured here and
was a no-op that looked like a knob — the general lesson being that a
score-modifying stage's position in the pipeline is part of its semantics.

**A uniform boost cannot target.** Multiplying all summaries by the same factor
shifts the whole summary population up, so what it actually buys is visibility
for whichever *kind* of group is most numerous — usually the small, low-level
groups produced in bulk by a partition, not the coarse ones a broad question
wants. It also breaks the comparability of the score scale: retrieval scores
across heterogeneous unit types are not calibrated to begin with, and rescaling
one type by hand is a guess about a calibration that was never measured.

The targeted alternative is `retrieval.summary_scope = drill-down`, which
removes the competition instead of re-weighting it: summaries are ranked only
against other summaries, then expanded.

## Why RAG architectures have this knob
Because a heterogeneous index (leaves plus summaries, or text plus tables) always
raises the question of how to compare scores across unit types, and a
multiplicative prior is the simplest available answer. It is here so the simple
answer can be measured — and so the measurement can show that the structural fix
is better.

## When it is useful
- **Diagnostically**, to confirm that summaries are being outvoted rather than
  being genuinely poor matches.
- **Mildly**, on a hierarchy whose groups are uniform in size, where "boost all
  summaries" is close to "boost the ones I mean".
- **Prefer `drill-down`** whenever the goal is to have broad questions answered
  from broad units.

## Interactions
Requires `index.hierarchy`, and a `retrieval.summary_scope` that includes
summaries at all — `drill-down` there is the targeted alternative to this
knob, and usually the better one. Amplifies whatever `index.min_group` let
through, so a low floor plus a boost is the configuration most likely to fill
the top-k with near-duplicate summaries; `retrieval.summary_levels` is the way
to exclude a level rather than re-weight it.
