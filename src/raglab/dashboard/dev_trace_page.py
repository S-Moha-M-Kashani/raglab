"""The developer's trace page: every step the widget's model took on a thread.

`GET /dev/trace?key=…` lists the threads; `&thread=…` shows one, step by step
— the system lines the model was handed, the reader's question, each tool
call with its arguments, the tool's reply, and the answer, with a token account
where the model reported one. It is a checkout window for a developer asking
"why did it say that?", not a reader surface: it answers to one key from the
environment (`RAGLAB_DEV_KEY`) and is a 404 otherwise — a 403 would already
have said the page exists. Plain HTML built here, no script, no theme: it
reads the conversation log and writes nothing.
"""
from __future__ import annotations

import html
import json
import os

from raglab.agents import widget

KEY_ENV = 'RAGLAB_DEV_KEY'

_STYLE = """
body{font:14px/1.5 -apple-system,system-ui,sans-serif;max-width:60em;margin:2em auto;padding:0 1em;color:#222;background:#fff}
h1{font-size:1.2em}a{color:#0645ad}
.step{border-left:4px solid #999;margin:1em 0;padding:.4em .8em;background:#f6f6f6}
.system{border-color:#888}.human{border-color:#1a7f37}.ai{border-color:#0969da}.tool{border-color:#bf8700}
.label{font-weight:600;font-size:.85em;text-transform:uppercase;letter-spacing:.05em;color:#555}
pre{white-space:pre-wrap;word-break:break-word;margin:.3em 0;font-size:.9em}
.meta{color:#666;font-size:.85em}
.trimmed{opacity:.55}
summary{cursor:pointer}
"""


def allowed(key: str) -> bool:
    """True only when a key is configured and this is it."""
    expected = (os.environ.get(KEY_ENV) or '').strip()
    return bool(expected) and (key or '').strip() == expected


def _esc(value) -> str:
    return html.escape(str(value if value is not None else ''))


def _page(title: str, body: str) -> str:
    return (f'<!doctype html><meta charset="utf-8"><title>{_esc(title)}</title>'
            f'<style>{_STYLE}</style><h1>{_esc(title)}</h1>{body}')


def index(key: str) -> str:
    names = widget.threads()
    items = ''.join(
        f'<li><a href="/dev/trace?key={_esc(key)}&thread={_esc(n)}">{_esc(n)}</a></li>'
        for n in names) or '<li class="meta">no conversations yet</li>'
    return _page('widget traces', f'<p class="meta">{len(names)} thread(s), '
                                  f'newest first.</p><ol>{items}</ol>')


def _step(number: int, step: dict, trimmed: bool = False) -> str:
    kind = step['kind']
    parts = [f'<div class="step {kind}{" trimmed" if trimmed else ""}">'
             f'<div class="label">{number}. {_esc(kind)}'
             + (' · outside the window' if trimmed else '')]
    if kind == 'tool':
        parts.append(f' · {_esc(step.get("name"))} → {_esc(step.get("tool_call_id"))}')
    parts.append('</div>')
    if step.get('text'):
        parts.append(f'<pre>{_esc(step["text"])}</pre>')
    for call in step.get('tool_calls') or []:
        parts.append(f'<div class="meta">calls <b>{_esc(call.get("name"))}</b> '
                     f'({_esc(call.get("id"))})</div>'
                     f'<pre>{_esc(json.dumps(call.get("args"), ensure_ascii=False, indent=2))}</pre>')
    if step.get('input_tokens') is not None or step.get('output_tokens') is not None:
        parts.append(f'<div class="meta">tokens in {_esc(step.get("input_tokens"))} · '
                     f'out {_esc(step.get("output_tokens"))}</div>')
    parts.append('</div>')
    return ''.join(parts)


def _standing(steps: list) -> str:
    """What the model stands on before it reads a step: the system prompt
    `create_agent` binds to every call (not in the log, so shown from the
    fixture), the tools it may call, and the window `trim_and_call` applies."""
    rest = [s for s in steps if s['kind'] != 'system']
    dropped = max(0, len(rest) - widget.MAX_HISTORY)
    tools = ', '.join(t.name for t in widget.TOOLS)
    return (
        '<details><summary class="label">standing system prompt (every call)'
        '</summary><pre>' + _esc(widget.SYSTEM_PROMPT) + '</pre></details>'
        f'<p class="meta">tools it may call: {_esc(tools)}</p>'
        f'<p class="meta">window: every system line below, plus the last '
        f'{widget.MAX_HISTORY} other messages — '
        + (f'the {dropped} oldest non-system step(s) below are no longer sent.'
           if dropped else 'nothing has been trimmed yet.')
        + f' Tool hops per turn stop at {widget.hooks.MAX_TOOL_HOPS}; a question is '
        f'capped at {widget.MAX_QUESTION} characters.</p>')


def thread(key: str, name: str) -> str:
    found = widget.trace(name)
    steps = found['steps']
    head = (f'<p class="meta"><a href="/dev/trace?key={_esc(key)}">← all threads</a>'
            f' · experiment {_esc(found["experiment_id"]) or "—"}'
            f' · dataset {_esc(found["dataset_id"]) or "—"}'
            f' · began {_esc(found["started_at"]) or "—"}'
            f' · {len(steps)} step(s)</p>')
    rest = [s for s in steps if s['kind'] != 'system']
    outside = {id(s) for s in rest[:max(0, len(rest) - widget.MAX_HISTORY)]}
    body = ''.join(_step(i, s, trimmed=id(s) in outside)
                   for i, s in enumerate(steps, 1)) \
        or '<p class="meta">this thread holds no messages.</p>'
    return _page(f'trace · {found["thread"]}', head + _standing(steps) + body)
