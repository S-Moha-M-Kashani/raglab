# Security policy

The RAG lab is a single-user workbench that runs on the machine in front of you
and talks to model providers on your behalf. This file says what is defended,
what is not, and how to tell me about a hole in either.

## Supported version

The newest release point on `master` — the tag `pyproject.toml` names. There are
no maintained older branches, so a fix lands as the next release and nothing is
back-ported.

## The boundary, in one paragraph

`uv run --extra local-embeddings raglab` serves the lab on `127.0.0.1:9002`: the
panel at `/`, the read-only Inspector at `/inspector`, the board at
`/leaderboard`. It is loopback because that is uvicorn's default and nothing
here overrides it. Under Docker the container has to listen on `0.0.0.0`, so
reachability is decided on the host side instead, and `compose.yaml` publishes
the port as `127.0.0.1:9002:9002` for exactly that reason. **There is no login
and no separation between users** — every visitor to that port is the operator.
A page you type an API key into does not belong on a LAN, in a tunnel, or behind
a reverse proxy open to anyone else.

## Where a key lives

An OpenRouter key typed into the panel's Settings is held in **process memory
only** — no file, no environment variable, no log line, no run file, no ledger
row, no archive. It dies with the process. That is a tested property, not an
intention: `llm_backends/tests/test_credentials.py` and the widget's own tests
go looking for the key in everything the lab writes. A key placed in `.env`
instead is an ordinary file on disk with ordinary file permissions, and the lab
makes no further promise about it.

The developer's step-by-step trace page at `/dev/trace` does not exist unless
`RAGLAB_DEV_KEY` is set — the route answers 404 — and its key is typed on the
page rather than passed in the address bar, so it does not land in browser
history.

## What is not defended, and is not claimed to be

- **There is no prompt-injection fence.** The lab exists to put corpus text in
  front of a model: retrieved chunks reach the answerer, and the four judged
  metrics reach a judge. Text you index can steer what a model says about it.
  Index a corpus you are willing to hand to a model, and read a number produced
  over hostile text as evidence about that text, not as a verdict.
- **The durable records are plain local files** — `.runs/`, `databases/*.db`,
  `.datasets/`. Nothing is encrypted at rest, and all of them are git-ignored
  so they never travel with the repository.
- **The bundled corpora are synthetic.** They are fixtures written for this
  project, not anybody's diary, and they carry no personal data to leak.

## Reporting a vulnerability

Email **s.moha.m.kashani@gmail.com** with a subject beginning `raglab
security`. Please include what you did, what happened, and the version or commit
you saw it on. Do not open a public issue for anything that could be used
against a running copy.

This is a personal project with one maintainer: expect a reply within a week,
and no bounty — there is no money behind this, only my thanks and credit in the
release note if you want it.
