# run.ragas_mode — which RAGAS metrics are computed

- **Step:** Run control. **Values:** `offline`, `judged`, `off`.

## What the knob does
`offline` scores the retrieved context against the ground-truth quotes with
string similarity — no model, no key, no variance. `judged` adds the five
model-graded metrics: faithfulness, answer relevancy, factual correctness, and
judged context precision and recall. **Four of those five are what a
configuration is actually chosen by.** `off` skips RAGAS entirely.

## What it means scientifically
The three modes are three different measurement instruments, and the difference
is not merely cost:

- **Offline metrics are deterministic and cheap, and they measure a proxy.**
  String overlap between retrieved context and known-good quotes is a real
  signal — it detects whether the right text was retrieved — but it cannot see
  paraphrase, cannot judge whether an answer is *supported*, and cannot compare
  an answer to a reference in another language. Its virtue is zero variance: the
  same run gives the same number, so it is the right instrument while iterating
  on retrieval.
- **Judged metrics measure the thing you care about, with an error term.** RAGAS
  defines faithfulness and relevancy by prompted decomposition and verification,
  so the metric is only as good as the judge and carries genuine
  run-to-run variance. That is why the lab reports `decision_score()` — the
  unweighted mean of exactly four judged metrics, `None` unless all four are
  present — **never without `decision_spread()`**: a mean without a spread
  invites a ranking that the error bars do not support.
- **`off`** is for runs where only the deterministic pipeline behaviour is under
  test, and it keeps the comparison honest by not producing a metric nobody
  graded.

## Why RAG architectures have this knob
Because RAG evaluation has two regimes: fast, deterministic feedback while
tuning the parts, and expensive, judged measurement when deciding. Conflating
them either makes iteration unaffordable or makes decisions unsupported.

## When each option is useful
- **`offline`** while sweeping index and retrieval knobs — free, fast, and
  sensitive to exactly what those knobs change.
- **`judged`** for any decision, any leaderboard row meant to be compared, and
  any claim about answer quality.
- **`off`** for smoke runs, plumbing tests and cost-bounded reruns.

## Interactions
`generation.ragas_model` names the judge; `run.ragas_limit` bounds how many
questions are graded; the four deciding metrics are what `group()`/`verdict()`
rank a sweep by, and the board records the judge as a column because rows graded
by different judges are not comparable.
