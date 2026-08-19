"""Live probes that live inside the package, not in tests/: tests/conftest.py
pins the suite offline and blanks the developer's keys, and everything in this
folder exists to do what that plumbing forbids — pay for real model calls.
Every test here skips unless its file is named on the pytest command line."""
