# run.openrouter_key — the key the OpenRouter backend calls with

- **Step:** Run control. **Held in process memory only.**

## What the knob does
Lets a lab that is already running reach a remote model, without restarting it
to pick up an environment variable. The key is held in the lab process and
written **nowhere** — not to a run file, not to the experiment ledger, not to
the browser — so it is forgotten when the lab stops. `OPENROUTER_API_KEY` in the
environment is still how a lab *starts* with one. Setting the key does not change
which backend runs: that is `run.mode`, and a model on this machine needs no key
at all.

## What it means scientifically — or rather, as engineering
This is not a measurement knob; it is a **security boundary**, and it is
documented beside the knobs because getting it wrong contaminates artefacts
rather than numbers:

- **Durable artefacts are shareable artefacts.** This lab's run files, ledger
  rows and archives are meant to be copied, published and imported elsewhere. A
  secret that reaches any of them is disclosed by the very act of sharing
  evidence, which is why the tests actually grep for the key across artefacts.
- **Process memory is the narrowest lifetime available** for a credential that
  must be usable by a running server: no file to leak, no env var inherited by
  child processes, no browser storage, and an automatic end at shutdown.
- **Separating credential from backend selection** avoids the classic
  ambiguity where providing a key silently changes which provider runs. Here the
  provider is chosen explicitly and the key only enables it.

## Why RAG architectures have this knob
Because remote judges and answerers cost money and require credentials, while
the lab must also run fully offline. Entering the key at the panel keeps the two
worlds in one process without pushing a secret into configuration files that get
committed, copied or exported.

## When it is useful
- **A running lab** that needs to reach a remote model for one experiment.
- **Shared or demo machines**, where nothing should persist after the process
  exits.
- **Not for unattended runs**: a lab that must start with a key should get it
  from the environment.

## Interactions
Enables `run.mode = OpenRouter` and the remote model catalogues (which offer
only models verified reachable on the account). Never appears in `.runs/`, the
ledger, an archive, or the widget's conversation log.
