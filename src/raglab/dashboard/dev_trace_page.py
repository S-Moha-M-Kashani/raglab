"""The developer's trace page: every step the widget's model took on a thread.

`GET /dev/trace` lists the threads; `?thread=…` shows one, step by step — the
system lines the model was handed, the reader's question, each tool call with
its arguments, the tool's reply, and the answer, with a token account where the
model reported one. It is a checkout window for a developer asking "why did it
say that?", not a reader surface.

It answers to one key from the environment (`RAGLAB_DEV_KEY`) and is a 404
without one — a 403 would already have said the page exists. With a key
configured, the page asks for it in a masked field: the key travels once, in a
POST body, and never in the address bar, the history, a link or a log. What
unlocks the browser afterwards is a session cookie holding a random token this
process remembers in memory — the same regime as the panel-typed OpenRouter
key: no file, no env, no artifact. A restart forgets every token, which for a
developer's window is the right default.

Plain HTML built here, no script beyond the theme stamp the three surfaces
share; it reads the conversation log and writes nothing.
"""
from __future__ import annotations

import hmac
import html
import json
import os
import secrets
from urllib.parse import quote

from raglab.agents import widget

COOKIE = 'raglab_dev'
PATH = '/dev/trace'

# Tokens issued to browsers that typed the key. Process memory only.
_TOKENS: set[str] = set()


def configured() -> bool:
    """True when a developer key is set at all — the page exists only then."""
    return bool((os.environ.get('RAGLAB_DEV_KEY') or '').strip())


def allowed(key: str) -> bool:
    """True only when a key is configured and this is it."""
    expected = (os.environ.get('RAGLAB_DEV_KEY') or '').strip()
    return bool(expected) and hmac.compare_digest((key or '').strip(), expected)


def issue_token() -> str:
    token = secrets.token_urlsafe(32)
    _TOKENS.add(token)
    return token


def revoke(token: str | None) -> None:
    _TOKENS.discard(token or '')


def unlocked(token: str | None) -> bool:
    return bool(token) and token in _TOKENS


def thread_href(name: str) -> str:
    return f'{PATH}?thread={quote(name, safe="")}'


def _esc(value) -> str:
    return html.escape(str(value if value is not None else ''))


# ── the page ────────────────────────────────────────────────────────────────
# Chassis, plate, card and the three faces all come from tokens.css, so Day
# and Night arrive by construction and this sheet holds no dark block of its
# own. The widget is a helper, not a stage, so no step ink appears here: the
# transcript reads as a tape — a mono rail of step numbers, the step's kind as
# the lab's small label, the content on a card. Who spoke is told by where the
# card sits: the reader flush left, the model one rail in, the tools folded.
_STYLE = """
html{background:var(--plate)}
body{margin:0;min-height:100vh;color:var(--ink);background:var(--plate);font:var(--t-base)/1.55 var(--sans)}
a{color:inherit}
.mast{background:var(--chassis);color:var(--chassis-ink);padding:var(--s-4) var(--gutter)}
.mast-inner{max-width:60rem;margin:0 auto;display:flex;align-items:baseline;gap:var(--s-4);flex-wrap:wrap}
.mast h1{margin:0;font:400 var(--t-xl)/1.1 var(--slab);color:#fff;letter-spacing:-.01em}
.mast h1 small{font:var(--label-font);letter-spacing:var(--label-track);text-transform:uppercase;color:var(--chassis-ink-soft);display:block;margin-bottom:var(--s-1)}
.mast .crumbs{flex-basis:100%;order:3;color:var(--chassis-ink-soft);font:var(--t-sm)/1.4 var(--mono);min-width:0;overflow-wrap:anywhere}
.mast .crumbs a{color:var(--chassis-ink)}
.mast form{margin-left:auto}
.lock{background:transparent;color:var(--chassis-ink);border:1px solid var(--chassis-ink-soft);border-radius:var(--radius-sm);padding:var(--s-1) var(--s-3);font:var(--t-sm)/1.4 var(--sans);cursor:pointer}
.lock:hover{background:var(--chassis-soft);color:#fff}
main{max-width:60rem;margin:0 auto;padding:var(--s-5) var(--gutter) var(--s-7)}
.lede{color:var(--ink-soft);margin:0 0 var(--s-5);max-width:48rem}
.meta{color:var(--ink-soft);font:var(--t-sm)/1.5 var(--mono)}
/* the index */
.threads{list-style:none;margin:0;padding:0;border-top:1px solid var(--rule)}
.threads li{display:grid;grid-template-columns:minmax(12rem,auto) 1fr;gap:var(--s-2) var(--s-5);padding:var(--s-3) 0;border-bottom:1px solid var(--rule)}
.threads a{font:400 var(--t-md)/1.3 var(--slab);text-decoration:none}
.threads a:hover{text-decoration:underline}
.threads .last{color:var(--ink);grid-column:2}
.threads .last:before{content:"“"}.threads .last:after{content:"”"}
.empty{padding:var(--s-6) 0;color:var(--ink-soft)}
/* what the model stands on */
.standing{background:var(--card);border:1px solid var(--rule);border-radius:var(--radius-md);padding:var(--s-3) var(--s-4);margin-bottom:var(--s-5);box-shadow:var(--shadow)}
.standing summary{font:var(--label-font);letter-spacing:var(--label-track);text-transform:uppercase;color:var(--ink-soft);cursor:pointer}
.standing p{margin:var(--s-2) 0 0}
/* the tape */
.tape{display:grid;grid-template-columns:3rem 1fr;column-gap:var(--s-3);row-gap:var(--s-3)}
.no{font:var(--t-sm)/1.6 var(--mono);color:var(--ink-soft);text-align:right;padding-top:var(--s-3);font-variant-numeric:tabular-nums}
.card{background:var(--card);border:1px solid var(--rule);border-radius:var(--radius-md);padding:var(--s-3) var(--s-4);box-shadow:var(--shadow);min-width:0}
.card .kind{font:var(--label-font);letter-spacing:var(--label-track);text-transform:uppercase;color:var(--ink-soft);display:flex;gap:var(--s-3);flex-wrap:wrap}
.card .kind .mono{font:var(--t-xs)/1.5 var(--mono);text-transform:none;letter-spacing:0}
.ai .card{margin-left:var(--s-6)}
.system .card{background:transparent;box-shadow:none;border-style:dashed}
.tool .card{margin-left:var(--s-6);padding:0}
.tool details{padding:var(--s-3) var(--s-4)}
.tool summary{cursor:pointer;list-style:none;display:flex;gap:var(--s-3);align-items:baseline;flex-wrap:wrap}
.tool summary::-webkit-details-marker{display:none}
.tool summary:before{content:"▸";color:var(--ink-soft);font:var(--t-sm)/1 var(--mono)}
.tool details[open] summary:before{content:"▾"}
pre{white-space:pre-wrap;overflow-wrap:anywhere;margin:var(--s-2) 0 0;font:var(--t-sm)/1.5 var(--mono)}
.card .calls{margin-top:var(--s-3);padding-top:var(--s-3);border-top:1px dashed var(--rule)}
.card .bill{margin-top:var(--s-2)}
.trimmed{opacity:.5}
.cut{grid-column:1/-1;display:flex;align-items:center;gap:var(--s-3);color:var(--ink-soft);font:var(--label-font);letter-spacing:var(--label-track);text-transform:uppercase}
.cut:before,.cut:after{content:"";flex:1;border-top:1px dashed var(--rule)}
/* the plate */
main.gate{max-width:none;margin:0;min-height:100vh;box-sizing:border-box;display:grid;place-items:center;background:var(--chassis);color:var(--chassis-ink);padding:var(--gutter)}
.plate{background:var(--card);color:var(--ink);border-radius:var(--radius-md);box-shadow:0 1px 2px rgba(0,0,0,.2),0 24px 48px -24px rgba(0,0,0,.8);padding:var(--s-6);width:min(22rem,100%)}
.plate h1{margin:0 0 var(--s-1);font:400 var(--t-xl)/1.1 var(--slab)}
.plate .sub{font:var(--label-font);letter-spacing:var(--label-track);text-transform:uppercase;color:var(--ink-soft);margin:0 0 var(--s-5)}
.plate label{display:block;font:var(--label-font);letter-spacing:var(--label-track);text-transform:uppercase;color:var(--ink-soft);margin-bottom:var(--s-1)}
.plate input{width:100%;box-sizing:border-box;font:var(--t-md)/1.4 var(--mono);letter-spacing:.2em;padding:var(--s-2) var(--s-3);border:1px solid var(--rule);border-radius:var(--radius-sm);background:var(--plate);color:var(--ink)}
.plate button{margin-top:var(--s-4);width:100%;padding:var(--s-2) var(--s-3);font:600 var(--t-base)/1.4 var(--sans);color:#fff;background:var(--chassis);border:1px solid var(--chassis);border-radius:var(--radius-sm);cursor:pointer}
.plate button:hover{background:var(--chassis-soft)}
.plate .err{color:var(--alert);margin:0 0 var(--s-3)}
.plate .hint{color:var(--ink-soft);font-size:var(--t-sm);margin:var(--s-4) 0 0}
:focus-visible{outline:2px solid var(--step-generation-lit);outline-offset:2px}
@media (max-width:40rem){.tape{grid-template-columns:1fr}.no{text-align:left;padding:0}.ai .card,.tool .card{margin-left:0}}
"""

_STAMP = ("<script>try{var t=localStorage.getItem('raglab-theme');"
          "if(t==='day'||t==='night')document.documentElement.dataset.theme=t}"
          "catch(e){}</script>")


def _head(title: str) -> str:
    return ('<!doctype html><html><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{_esc(title)}</title>' + _STAMP
            + '<link rel="stylesheet" href="/tokens.css">'
            f'<style>{_STYLE}</style></head><body>')


def _page(title: str, body: str) -> str:
    return _head(title) + body + '</body></html>'


def _mast(crumbs: str = '') -> str:
    return ('<header class="mast"><div class="mast-inner">'
            '<h1><small>RAG lab · developer</small>widget traces</h1>'
            + (f'<div class="crumbs">{crumbs}</div>' if crumbs else '')
            + f'<form method="post" action="{PATH}/lock">'
            '<button class="lock" type="submit">Lock</button></form>'
            '</div></header>')


def unlock_page(next_thread: str = '', error: str = '') -> str:
    """The plate: one masked field. `next_thread` is where the browser was
    heading; it is echoed only as a hidden field, never as a link."""
    return _page('unlock · widget traces',
                 '<main class="gate"><form class="plate" method="post" '
                 f'action="{PATH}" autocomplete="off">'
                 '<h1>RAG lab</h1><p class="sub">developer trace</p>'
                 + (f'<p class="err" role="alert">{_esc(error)}</p>' if error else '')
                 + '<label for="key">developer key</label>'
                 '<input id="key" name="key" type="password" autocomplete="off" '
                 'autofocus required>'
                 + (f'<input type="hidden" name="next" value="{_esc(next_thread)}">'
                    if next_thread else '')
                 + '<button type="submit">Unlock</button>'
                 '<p class="hint">The key is <code>RAGLAB_DEV_KEY</code> in the '
                 'server\'s <code>.env</code>. It is sent once and stays out of '
                 'the address bar.</p></form></main>')


def index() -> str:
    rows = widget.thread_summaries()
    items = ''.join(
        f'<li><a href="{_esc(thread_href(r["thread"]))}">{_esc(r["thread"])}</a>'
        f'<span class="meta">{r["questions"]} question{"" if r["questions"] == 1 else "s"}'
        f' · {r["steps"]} step{"" if r["steps"] == 1 else "s"}</span>'
        + (f'<span class="last">{_esc(r["last"][:120])}</span>' if r['last'] else '')
        + '</li>'
        for r in rows)
    body = (f'<ol class="threads">{items}</ol>' if rows else
            '<p class="empty">No conversations yet — ask the widget something '
            'and come back.</p>')
    return _page('widget traces', _mast() + '<main>'
                 f'<p class="lede">{len(rows)} conversation{"" if len(rows) == 1 else "s"}'
                 ' — one per experiment, plus <code>general</code>, newest first. '
                 'Every question a reader sends lands in its experiment\'s '
                 'conversation; open one to read each prompt step by step.</p>'
                 + body + '</main>')


def _step(number: int, step: dict, trimmed: bool = False) -> str:
    kind = step['kind']
    cls = f'{kind}{" trimmed" if trimmed else ""}'
    no = f'<div class="no {cls}">{number:02d}</div>'
    text = f'<pre>{_esc(step["text"])}</pre>' if step.get('text') else ''
    bill = ''
    if step.get('input_tokens') is not None or step.get('output_tokens') is not None:
        bill = (f'<div class="meta bill">tokens in {_esc(step.get("input_tokens"))}'
                f' · out {_esc(step.get("output_tokens"))}</div>')
    if kind == 'tool':
        return (no + f'<div class="tool {cls}"><div class="card"><details>'
                f'<summary><span class="kind">tool reply <span class="mono">'
                f'{_esc(step.get("name"))} → {_esc(step.get("tool_call_id"))}'
                f'</span></span></summary>{text}</details></div></div>')
    calls = ''
    for call in step.get('tool_calls') or []:
        calls += (f'<div class="calls"><div class="kind">calls <span class="mono">'
                  f'{_esc(call.get("name"))} ({_esc(call.get("id"))})</span></div>'
                  f'<pre>{_esc(json.dumps(call.get("args"), ensure_ascii=False, indent=2))}</pre></div>')
    label = {'human': 'reader', 'ai': 'model', 'system': 'system'}.get(kind, kind)
    return (no + f'<div class="{cls}"><div class="card"><div class="kind">{_esc(label)}'
            '</div>' + text + calls + bill + '</div></div>')


def _standing(steps: list) -> str:
    """What the model stands on before it reads a step: the system prompt
    `create_agent` binds to every call (not in the log, so shown from the
    fixture), the tools it may call, and the window `trim_and_call` applies."""
    rest = [s for s in steps if s['kind'] != 'system']
    dropped = max(0, len(rest) - widget.MAX_HISTORY)
    tools = ', '.join(t.name for t in widget.TOOLS)
    return (
        '<details class="standing"><summary>standing system prompt — every call'
        '</summary><pre>' + _esc(widget.SYSTEM_PROMPT) + '</pre>'
        f'<p class="meta">tools it may call: {_esc(tools)}</p>'
        f'<p class="meta">window: every system line below, plus the last '
        f'{widget.MAX_HISTORY} other messages — '
        + (f'the {dropped} oldest non-system step(s) are no longer sent.'
           if dropped else 'nothing has been trimmed yet.')
        + f' Tool hops per turn stop at {widget.hooks.MAX_TOOL_HOPS}; a question is '
        f'capped at {widget.MAX_QUESTION} characters.</p></details>')


def thread(name: str) -> str:
    found = widget.trace(name)
    steps = found['steps']
    crumbs = (f'<a href="{PATH}">all threads</a> / {_esc(found["thread"])}'
              f' · experiment {_esc(found["experiment_id"]) or "—"}'
              f' · dataset {_esc(found["dataset_id"]) or "—"}'
              f' · began {_esc(found["started_at"]) or "—"}'
              f' · {len(steps)} step(s)')
    rest = [s for s in steps if s['kind'] != 'system']
    outside = {id(s) for s in rest[:max(0, len(rest) - widget.MAX_HISTORY)]}
    parts, cut_drawn = [], False
    for i, s in enumerate(steps, 1):
        trimmed = id(s) in outside
        if not trimmed and outside and not cut_drawn and s['kind'] != 'system':
            parts.append('<div class="cut">from here on, sent to the model</div>')
            cut_drawn = True
        parts.append(_step(i, s, trimmed=trimmed))
    tape = (f'<div class="tape">{"".join(parts)}</div>' if steps else
            '<p class="empty">This thread holds no messages.</p>')
    return _page(f'trace · {found["thread"]}',
                 _mast(crumbs) + '<main>' + _standing(steps) + tape + '</main>')
