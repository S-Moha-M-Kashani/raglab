"""The lab, served for a browser test — the real entry point, redirected.

A real browser needs a real socket, so the browser suite cannot ride the
in-process `TestClient` the rest of the dashboard tests use. This module is
what the `lab_server` fixture runs in a child process, and its whole job is to
repeat inside that process the redirections `src/raglab/conftest.py` performs
in the parent: the three database paths arrive as environment variables and
need nothing done to them, while `RUNS_DIR` is a module attribute rather than
a variable anyone reads per call, so it is reassigned here on every module
that binds it.

Run as `python -m raglab.dashboard.tests.browser_lab_server`, with
`RAGLAB_BROWSER_PORT` and `RAGLAB_BROWSER_RUNS` naming the port to bind and
the directory run files may be written into. Nothing imports this module; the
fixture names it on a command line, the way uvicorn names the app.
"""
import os
from pathlib import Path

import uvicorn

#: The app string uvicorn is given, spelled exactly as `cli/serve.py` spells
#: it — a browser test that booted some other object would prove nothing about
#: the lab people actually launch.
APP = 'raglab.dashboard.served_lab:app'


def redirect_runs_dir(runs: Path) -> None:
    """Point every module that binds `RUNS_DIR` at a temporary directory.

    The three modules are the three the offline suite's own fixture patches,
    and `panel_server` is deliberately not among them: its only use of the name
    is `RUNS_DIR.relative_to(ROOT)` in the capability report, which a temporary
    directory outside the repo would turn into an error. Nothing is written
    through that binding, so it is left telling the truth about where a real
    lab would put its runs.

    The imports are inside the function so that this runs before the app is
    built: uvicorn imports `served_lab` when `run()` is called, and by then the
    modules are already in `sys.modules` with the attribute reassigned.
    """
    from raglab.agents.extra_tools import sweep
    from raglab.evaluation import leaderboard
    from raglab.evaluation import run_evaluation

    runs.mkdir(parents=True, exist_ok=True)
    for module in (run_evaluation, leaderboard, sweep):
        module.RUNS_DIR = runs


def main() -> None:
    redirect_runs_dir(Path(os.environ['RAGLAB_BROWSER_RUNS']))
    uvicorn.run(APP, host='127.0.0.1',
                port=int(os.environ['RAGLAB_BROWSER_PORT']),
                log_level='warning')


if __name__ == '__main__':
    main()
