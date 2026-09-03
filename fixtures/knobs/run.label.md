# run.label — what this run is called on the leaderboard

- **Step:** Run control. **Worth writing:** a row named "semantic-drift" tells
  you nothing three days later.

## What the knob does
Names the experiment. The label travels with the row into the ledger, the
leaderboard board and any archive exported from it.

## What it means scientifically
Not a measurement knob, but a knob about the **integrity of the record**, and it
carries more weight than it looks like:

- **A row must be interpretable without its author.** The board puts every
  experiment that touched one corpus in one table and names no winner, precisely
  because rows graded by different judges over different question sets are not
  ranked by the board. That makes the label the reader's handle for *why* this
  configuration was run — the hypothesis, not just the knob that moved.
- **Configuration is recorded automatically; intent is not.** Every knob value is
  on the row already, so a label repeating a knob value adds nothing. What cannot
  be recovered later is what the run was testing: "does drift chunking beat fixed
  at 500 chars", "control: leaves only", "rerun after judge screen".
- **Reproducibility has a human half.** Fingerprints and ledger rows make a run
  mechanically reproducible; the label is what makes a body of runs *readable* as
  an argument months later. In an accumulating record, an unlabelled row is
  effectively unfindable.

## Why RAG architectures have this knob
Because sweeping produces dozens of rows an hour and the interesting question is
almost never "which number is highest" but "what was I comparing". The label is
the cheapest possible experiment note, stored in the same place as the evidence.

## When it is useful
- **Always**, for any run that will be compared later.
- **State the hypothesis or the role**: name the control as a control, name the
  arm as an arm.
- **Include what the knobs cannot show**: the corpus condition you were
  suspicious of, the previous row this one responds to.

## Interactions
Recorded on the ledger row and the leaderboard board; carried into an exported
archive, which is preserved verbatim — so a label written today is what a reader
on another machine will see. Worth naming the knob under test and the sampling
it was measured under, because `run.limit`, `run.balance` and `run.labels`
decide which questions the row's means are over, and `index.dataset` decides
which table it lands in.
