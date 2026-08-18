"""The end-to-end check:

    uv run python -m raglab.widget [model] [question ...]

or this file itself, under a debugger. Real calls: a live model over the
network, or a real CLI process. Not a console entry point and not a test —
the suite is offline, and this is the thing that is not.

Nothing here catches or exits: a refusal should stop the debugger where it
was raised, and the two questions are the two tools.
"""
import sys
import time

if __package__ in (None, ''):
    # A debugger's run-this-file button executes `python .../__main__.py`,
    # where a relative import has no package to be relative to — so stepping
    # through this package died on line one. Naming the package here, before
    # the imports below, is what makes `python -m raglab.widget` and the
    # green arrow the same run.
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    __package__ = 'raglab.widget'

from ..settings import load_env_file
from . import hooks
from .backends import DEFAULT_MODEL, WIDGET_MODELS, ask
from .hooks import HOOK_LOG
from .tools import TOOLS

QUESTIONS = ('Which ports do the lab and the Inspector serve on?',
             'What is 174 - 167?')


def main(model: str = DEFAULT_MODEL, questions: tuple = QUESTIONS) -> None:
    hooks.HOOKS_VERBOSE = True           # hooks print as they fire
    load_env_file()                      # the route's server did this already
    kind, label = WIDGET_MODELS[model]
    print(f'{label}  [{kind}]  tools: {[t.name for t in TOOLS]}\n')
    for question in questions:
        HOOK_LOG.clear()
        started = time.perf_counter()
        answer = ask(question, model)
        print(f'\n  {question}\n  → {answer.strip()}'
              f'\n  {time.perf_counter() - started:.1f}s, '
              f'{len(HOOK_LOG)} hooks\n')


if __name__ == '__main__':
    # argv, not argparse: two optional positionals, and a debugger that runs
    # this file with none of them gets the defaults.
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL,
         (' '.join(sys.argv[2:]),) if len(sys.argv) > 2 else QUESTIONS)
