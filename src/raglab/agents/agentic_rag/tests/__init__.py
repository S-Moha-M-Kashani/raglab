"""The agentic loop's live probes, inside the package they probe — the same
locality the widget's real-call harness has in `widget/__main__.py` — and not
in tests/, whose conftest pins the suite offline and blanks the developer's
keys. Everything in this folder exists to do what that plumbing forbids: pay
for real model calls. Every test here skips unless its file is named on the
pytest command line."""
