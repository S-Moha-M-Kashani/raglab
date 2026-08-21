// The lab's LLM widget, on every surface. One definition: this file builds its
// own launcher and window into <body> rather than three pages carrying the same
// thirty lines of markup — the way lab.js already builds the footer rail and the
// theme control. Loads after lab.js, whose escapeHtml it uses.
//
// One route behind it (POST /api/widget) plus the two that read and reset the
// conversation. Nothing here writes a run, a ledger row or a number, which is
// the whole reason a helper may sit on a read-only surface at all.
//
// Wrapped the way archive_io.js and experiment_handoff.js are wrapped, and for
// a harder reason than tidiness: this file and panel.js are two classic scripts
// sharing one global scope, and both need a `$`. A second top-level `const $`
// is not a shadowed name, it is a SyntaxError that stops panel.js loading at
// all — so the widget keeps its helpers to itself and publishes one object.
(() => {

  // --- the markup, built here ------------------------------------------------
  // Three pages, one definition. A surface gains the helper by loading this
  // file and nothing else, which is what stops the same thirty lines of markup
  // from drifting apart in three places.

  function mountWidget() {
    const launcher = document.createElement('button');
    launcher.id = 'widget-launch';
    launcher.className = 'widget-launch';
    launcher.type = 'button';
    launcher.title = 'Ask about this lab';
    launcher.textContent = '✳ Ask';

    const win = document.createElement('div');
    win.id = 'widget-window';
    win.className = 'widget-window';
    win.hidden = true;
    // Static author-written markup, so innerHTML carries nothing a reader typed.
    // The grips come last on purpose: the two focusable ones then follow the
    // input in document order, and where they sit on screen is CSS's business.
    win.innerHTML = `
  <div class="widget-head">
    <span id="widget-name" class="widget-name" role="button" tabindex="0" title="Ask about this lab">Lab helper</span>
    <span class="widget-head-actions">
      <button id="widget-settings" class="widget-close" type="button" aria-label="Settings" title="Choose the model">⚙</button>
      <button id="widget-close" class="widget-close" type="button" aria-label="Close">×</button>
    </span>
  </div>
  <div id="widget-config" class="widget-config" hidden>
    <label>model <select id="widget-model"></select></label>
  </div>
  <div id="widget-log" class="widget-log"></div>
  <form id="widget-form" class="widget-form">
    <input id="widget-input" type="text" placeholder="Ask about this project…" autocomplete="off">
    <button id="widget-send" type="submit">Send</button>
  </form>
  <div class="widget-grip widget-grip-corner" data-grip="corner" aria-hidden="true"></div>
  <div class="widget-grip widget-grip-top" data-grip="top" role="separator"
       aria-orientation="horizontal" tabindex="0"
       aria-label="Resize the helper — up and down change its height"></div>
  <div class="widget-grip widget-grip-left" data-grip="left" role="separator"
       aria-orientation="vertical" tabindex="0"
       aria-label="Resize the helper — left and right change its width"></div>`;

    document.body.append(launcher, win);
  }

  // --- the four helpers this file needs --------------------------------------
  // Copies of panel.js's, not imports of them: the widget rides on surfaces
  // panel.js is not on, so borrowing a name from it would make the helper work
  // on the Laboratory and be dead everywhere else. A few lines each, and they
  // are the whole dependency — escapeHtml is the one thing still borrowed, and
  // it comes from lab.js, which every surface loads first.

  const $ = (id) => document.getElementById(id);

  async function api(path, body, method = body ? 'POST' : 'GET') {
    const options = method === 'GET' ? undefined : {
      method,
      headers: body ? { 'Content-Type': 'application/json' } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    };
    const res = await fetch(path, options);
    const data = await res.json().catch(() => ({ detail: res.statusText }));
    if (!res.ok) throw new Error(data.detail || res.statusText);
    return data;
  }

  function saved(key) {
    try {
      return JSON.parse(localStorage.getItem(key) || 'null');
    } catch (e) {
      // A key someone hand-edited, or written by an older shape of this page.
      // Forgetting it is right; refusing to boot over it is not.
      return null;
    }
  }

  function remember(key, value) {
    try {
      if (value === null) localStorage.removeItem(key);
      else localStorage.setItem(key, JSON.stringify(value));
    } catch (e) { /* private browsing, a full quota: not worth a broken panel */ }
  }

  // --- which conversation this is ---------------------------------------------
  // One thread per experiment: open abc123 and you are in the conversation you
  // had about abc123, whenever that was, because widget.db kept it. With nothing
  // open there is one general thread shared by all three surfaces — not one per
  // page, which is exactly the reset this replaced.
  //
  // The key is read on every surface, which only works because the Laboratory,
  // the Inspector and the Leaderboard are one origin now. localStorage cannot
  // cross origins, and this is the thing that could not have travelled to :9003.
  const ACTIVE_EXPERIMENT = 'raglab-active-experiment';
  const GENERAL_THREAD = 'general';

  function widgetThread() {
    // Storage throws rather than returning null in a browser set to block
    // site data. This is a plain getter called on every keystroke's worth of
    // state check, not a one-time boot step, so there is no honest place from
    // in here to say so *once* — the only truthful options are silence or a
    // message on every single call, and the second is worse than the first.
    // So a reader in that browser lands on the general thread for as long as
    // the page stays open, quietly, rather than being told about a warning
    // that was never actually shown.
    try {
      return (localStorage.getItem(ACTIVE_EXPERIMENT) || '').trim() || GENERAL_THREAD;
    } catch (error) {
      return GENERAL_THREAD;
    }
  }

  function widgetAbout(experimentId) {
    const id = (experimentId || '').trim();
    try {
      if (id) localStorage.setItem(ACTIVE_EXPERIMENT, id);
      else localStorage.removeItem(ACTIVE_EXPERIMENT);
    } catch (error) { /* see widgetThread */ }
    // widgetName and widgetDraw are declared below this slice, in code that
    // never runs inside widget_thread.test.js — that contract pulls only the
    // text between these two markers into a bare vm context with no widget
    // mounted, to check the storage half alone. The guards are what let the
    // real page call both after mount while the sandboxed test calls neither.
    if (typeof widgetName === 'function') widgetName();
    if (typeof widgetDraw === 'function') widgetDraw();
  }

  // Which draw is newest, and whether a given thread is still the one on
  // screen — the two questions every redraw and every note has to answer
  // before it is allowed to touch the log. Both are pure functions of state
  // already declared in this slice (`DRAW_SEQ`, `widgetThread()`), which is
  // exactly what lets them live here rather than beside the DOM-touching code
  // that calls them: `widgetDrawThread` needs a real page to fetch and paint,
  // but deciding whether a fetch that already came back is still worth
  // painting needs nothing but arithmetic and a string compare.
  //
  // `DRAW_SEQ` counts every draw ever started, in the order `nextGeneration`
  // was called — synchronously, before that draw does any awaiting, so two
  // draws started back to back are numbered in the order they were *started*
  // regardless of which one's network round trip happens to finish first.
  let DRAW_SEQ = 0;

  function nextGeneration() {
    return ++DRAW_SEQ;
  }

  // True once some later draw has started — the generation this draw was
  // given is no longer the newest, so painting now would show this draw's
  // thread under whatever the newer one's header already reads.
  function supersedes(mine) {
    return mine !== DRAW_SEQ;
  }

  // True while the thread named at some earlier moment is still the thread
  // `widgetThread()` reports right now. A note captures this once, when it is
  // written, and checks it again just before it speaks — the gap between the
  // two is exactly the redraw it waited through.
  function stillCurrent(intended) {
    return widgetThread() === intended;
  }
  // --- end of the thread half -------------------------------------------------

  // The header says which conversation is on screen, because a helper that
  // quietly showed a different memory than the one you expect is worse than one
  // that shows none. Clicking it leaves the experiment: without that, a reader
  // who opened an experiment once has no way back to the general thread.
  function widgetName() {
    const thread = widgetThread();
    const general = thread === GENERAL_THREAD;
    const el = $('widget-name');
    el.textContent = general ? 'Lab helper' : `About ${thread}`;
    // The general thread has nothing to leave, so the title — and, on the
    // general thread, the point of clicking at all — has to change with it:
    // a button that always advertises "leave this experiment" while sitting
    // over the shared thread is offering an exit from a room you are not in.
    el.title = general ? 'Ask about this lab' : "Leave this experiment's conversation";
  }

  mountWidget();
  widgetName();

  $('widget-name').addEventListener('click', () => widgetAbout(''));
  $('widget-name').addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); widgetAbout(''); }
  });

  // Another surface just switched which experiment is open. It already
  // updated `localStorage` itself through `Widget.about`, and a `storage`
  // event never reaches the tab that made the write — only every *other*
  // open tab hears it, which is exactly the gap: with the Laboratory and the
  // Inspector both open on this experiment, the point of this whole feature,
  // leaving one tab's header and log stale would mean the screen names one
  // thread while `widgetThread()` — read fresh, from storage, at ask time —
  // already posts to another.
  window.addEventListener('storage', (event) => {
    if (event.key !== ACTIVE_EXPERIMENT) return;
    widgetName();
    widgetDraw();
  });

  // Replies are model output rendered into the page, so they pass through the
  // shared escapeHtml like every other untrusted string.

  function widgetSay(kind, text) {
    const log = $('widget-log');
    // The empty state goes on the first thing said, not on the first reply: a
    // panel of examples sitting above your own question reads as a menu you
    // failed to use. A `note` is not a turn in the conversation — it is the lab
    // saying what it just did, and it can arrive at a widget the reader has never
    // opened — so it lands under the examples instead of clearing them.
    const offer = log.querySelector('.widget-empty');
    if (offer && kind !== 'note') offer.remove();
    log.insertAdjacentHTML('beforeend',
      `<div class="widget-msg ${kind}">${escapeHtml(text)}</div>`);
    log.scrollTop = log.scrollHeight;
  }

  // --- what you can ask ------------------------------------------------------
  // The four examples come from the served fixture (fixtures/prompts/widget.yaml,
  // through the /api/widget response the model list already rides), because
  // clicking one sends that exact string to the model and model-facing text is a
  // fixture in this project.
  let WIDGET_STARTERS = [];

  // A question about the run currently on the Readings card, and empty when
  // there is none. The page that holds a run hands it over through
  // `Widget.offer`; nothing in here goes looking for one. Deliberately the
  // *lab's* last run and not the last conversation: the run chip is this
  // page's own suggestion, not a claim about what the thread holds — that
  // claim is `widgetDrawThread`'s, drawn from the lab on every open.
  let WIDGET_RUN_ASK = '';

  // Rendered only while the log is empty. Called again when a run lands, so a
  // widget left open and untouched picks up the chip for the run just finished.
  function widgetOffer() {
    const log = $('widget-log');
    // A note is not a conversation, so a log holding only notes is still a log
    // nobody has asked anything in — and the reader whose widget was opened *by*
    // a note is exactly the reader who has not seen the examples yet.
    if (log.querySelector('.widget-msg:not(.note)')) return;
    if (!WIDGET_STARTERS.length && !WIDGET_RUN_ASK) return;
    const chip = (text, extra = '') =>
      `<button type="button" class="widget-starter${extra}">${escapeHtml(text)}</button>`;
    const standing = log.querySelector('.widget-empty');
    if (standing) standing.remove();
    // Inserted rather than assigned over the log, which would take any note with
    // it — the offer is rebuilt whenever a run lands, and by then a note may
    // already be sitting there.
    log.insertAdjacentHTML('afterbegin',
      '<div class="widget-empty"><b>What you can ask</b>'
      + WIDGET_STARTERS.map((text) => chip(text)).join('')
      + (WIDGET_RUN_ASK ? chip(WIDGET_RUN_ASK, ' widget-starter-run') : '')
      + '</div>');
  }

  // A line the lab wrote, in the widget's log — which is where a reader finds it
  // on whichever surface the widget is on. Never `bot`: the model did not say
  // this, and a page borrowing the model's voice for its own statements is the
  // same lie a row telling you the wrong model produced it would be, one seam
  // earlier. A closed widget is opened, because a notice nobody can see is not a
  // notice; focus is deliberately left alone, since the click that caused this
  // happened on another surface and the caret is somewhere the reader put it.
  //
  // The note must land on the thread it is about, drawn from the lab, not on
  // whatever the log happened to hold a moment before. `openHandedExperiment`
  // calls `Widget.about` immediately before this, which starts a redraw that
  // wipes the whole log and rebuilds it from history the moment it resolves —
  // history that has never heard of this note, since a note is never sent to
  // the model. Appending first and drawing after would show the note for one
  // frame and then erase it.
  //
  // So this waits for a draw rather than racing one — but it must wait on
  // the *same* draw `Widget.about` already started, not fire a second GET of
  // the same history: two independent connections resolve in no guaranteed
  // order, so a second draw chained only to itself would not close the race,
  // it would just add a second way to lose it. `drawnFor` says which thread
  // the in-flight (or last-finished) draw actually covers; when that already
  // matches, this reuses it. It matches only when nothing has drawn this
  // thread yet — the widget was never opened, and the handoff failed before
  // `Widget.about` ran — that a fresh draw is started here.
  //
  // `.catch` guards the wait itself: a draw that rejects must not silently
  // swallow the note along with it — the note is said either way, because a
  // note that vanishes without a trace is worse than one shown a beat late.
  //
  // The final check closes one more face of the same problem: two
  // board-opens in quick succession start two draws, and if the reader is
  // already on the second experiment by the time the first note's draw
  // settles, appending that note now would glue "about A" text under B's
  // freshly drawn history and B's header. `supersedes` is the right test —
  // *not* "is thread A still current", which was tried first and still has a
  // gap: leave A and come straight back to A while A's own first draw is
  // still in flight, and the thread name matches again even though a newer
  // draw for that same thread has since started and will still wipe
  // whatever this note just wrote. Generation, not thread name, is what a
  // note actually needs to have survived — `stillCurrent` is kept alongside
  // it as a second, cheap check on the same fact for the same reason the
  // project never settles for one signal where a lie about provenance is
  // possible. A note that fails either check is dropped, not resurrected on
  // a later switch back — it is a one-off announcement of something that
  // just happened, not part of the conversation, so an announcement about a
  // moment the reader has since moved past has nothing left to say truthfully.
  function widgetNote(text) {
    const win = $('widget-window');
    if (win.hidden) win.hidden = false;
    const intended = widgetThread();
    if (drawnFor !== intended) widgetDraw();
    const mine = drawnGeneration;
    widgetDrawing.catch(() => {}).then(() => {
      if (supersedes(mine) || !stillCurrent(intended)) return;
      widgetSay('note', text);
    });
  }

  // Delegated, because the offer is rebuilt whenever a run lands.
  $('widget-log').addEventListener('click', (event) => {
    const starter = event.target.closest('.widget-starter');
    if (!starter) return;
    widgetAsk(starter.textContent);
  });

  async function widgetAsk(message) {
    widgetSay('you', message);
    $('widget-send').disabled = true;
    try {
      const data = await api('/api/widget',
        { message, model: $('widget-model').value, thread: widgetThread() });
      widgetSay('bot', data.reply);
      // The token account, when the backend reported one — an unreported
      // account renders nothing rather than a made-up zero.
      if (data.input_tokens != null) {
        widgetSay('meta', `${data.input_tokens} in / ${data.output_tokens} out tokens`);
      }
    } catch (error) {
      widgetSay('err', error.message);
    } finally {
      $('widget-send').disabled = false;
      $('widget-input').focus();
    }
  }

  // The model list and the starters are served, not kept here — fetched once, on
  // the first open. Returns its promise rather than swallowing it, so a caller
  // that needs the options in place first — `widgetDrawThread` needs
  // `WIDGET_STARTERS` before it can decide whether to offer them — can wait
  // on it with `.then` instead of racing it.
  let widgetOptionsLoaded = false;
  async function widgetLoadOptions() {
    if (widgetOptionsLoaded) return;
    try {
      const data = await api('/api/widget');
      $('widget-model').innerHTML = data.models.map((m) =>
        `<option value="${escapeHtml(m.value)}"${m.value === data.default ? ' selected' : ''}>`
        + `${escapeHtml(m.label)}</option>`).join('');
      WIDGET_STARTERS = data.starters || [];
      widgetOptionsLoaded = true;
    } catch (error) {
      widgetSay('err', error.message);
    }
  }

  // The log is drawn from the lab, never from a copy kept here: what a reader
  // sees is what the model remembers. A thread with nothing in it draws the
  // starters, which is the honest rendering of a conversation not yet had.
  //
  // Two draws can be in flight at once — the reader can leave an experiment,
  // or open a different one, before the first draw's fetch has come back —
  // and the two are for different threads, over two independent connections
  // that carry no ordering guarantee between them. Painting whichever
  // happens to resolve last, unconditionally, would show one thread's turns
  // under the other's header the moment the slower fetch lands: the exact
  // thing this whole feature exists to prevent, one layer down from the note
  // race. `mine` is this draw's generation, handed in by `widgetDraw` — the
  // only caller — at the moment it was started; `supersedes` says whether a
  // newer one has begun since, in which case this one stands down instead of
  // painting over a screen that has moved on.
  async function widgetDrawThread(mine) {
    const log = $('widget-log');
    let read;
    try {
      read = await api('/api/widget/history?thread='
                       + encodeURIComponent(widgetThread()));
    } catch (error) {
      if (!supersedes(mine)) widgetSay('err', error.message);
      return;
    }
    if (supersedes(mine)) return;
    log.innerHTML = '';
    // A 200 whose body does not carry a `turns` array is a bug on the lab's
    // side, and the honest response to that is an error line, not a cheerful
    // empty thread with starter chips — showing "nothing here yet" when the
    // truth is "the lab answered something I could not read" is exactly the
    // kind of lie this project's rows are not allowed to tell, one layer up
    // from where that rule is usually stated. Still resolves normally rather
    // than throwing, though: a rejection here would leave `widgetDrawing`
    // permanently rejected and, with it, silently drop every note still
    // waiting on it (including "no knob was changed", the one notice whose
    // silent absence is most dangerous).
    if (!Array.isArray(read.turns)) {
      widgetSay('err', 'The lab answered, but this thread’s history came '
        + 'back in a shape this page could not read.');
      return;
    }
    for (const turn of read.turns) widgetSay(turn.role, turn.text);
    if (!read.turns.length) widgetOffer();
  }

  // The draw currently in flight, or the last one to finish — kept so anything
  // that must follow a redraw, rather than race it, has something to wait on.
  // `widgetNote` is the reason this exists: `Widget.about` starts a draw and a
  // note is meant to land right after it, and every caller of `widgetDrawThread`
  // goes through here instead so that promise is always the current one.
  // `drawnFor` records which thread that draw covers and `drawnGeneration`
  // its `mine`, so a caller can tell an in-flight draw worth reusing from a
  // stale one worth ignoring — and, for a note, worth waiting on by
  // generation rather than by thread name (see `widgetNote`).
  //
  // The generation is assigned here, synchronously, before `widgetLoadOptions`
  // is even asked to run — not inside `widgetDrawThread` after that promise
  // settles. Two draws started back to back must be numbered in the order
  // they were *started*; assigning the number only once each one's own
  // options-load happened to finish would let an unrelated network jitter in
  // that unrelated request scramble which draw counts as newer.
  let widgetDrawing = Promise.resolve();
  let drawnFor = null;
  let drawnGeneration = 0;
  function widgetDraw() {
    drawnFor = widgetThread();
    drawnGeneration = nextGeneration();
    const mine = drawnGeneration;
    widgetDrawing = widgetLoadOptions().then(() => widgetDrawThread(mine));
    return widgetDrawing;
  }

  $('widget-launch').addEventListener('click', () => {
    const win = $('widget-window');
    widgetSetOpen(win.hidden);
    if (!win.hidden) {
      widgetDraw();
      $('widget-input').focus();
    }
  });

  $('widget-settings').addEventListener('click', () => {
    const row = $('widget-config');
    row.hidden = !row.hidden;
  });

  $('widget-close').addEventListener('click', () => { widgetSetOpen(false); });

  $('widget-form').addEventListener('submit', (event) => {
    event.preventDefault();
    const message = $('widget-input').value.trim();
    if (!message) return;
    $('widget-input').value = '';
    widgetAsk(message);
  });

  // --- resizing from the top and the left ------------------------------------
  // The window is anchored bottom-right, so growing it means gaining width and
  // height while the anchor holds — which is what dragging those two edges
  // outward looks like. The size is a preference, not a gesture, so it is
  // remembered under the same `lodestar:` prefix as the settings and the last run.
  const SAVED_WIDGET_SIZE = 'lodestar:raglab-widget-size';

  // Open or shut travels with the reader, the way the size already does. The
  // point of the helper being on three surfaces is that crossing between them
  // changes nothing about it — a window that shut itself on every navigation
  // would make the shared conversation pointless.
  const SAVED_WIDGET_OPEN = 'raglab-widget-open';

  function widgetSetOpen(open) {
    $('widget-window').hidden = !open;
    try { localStorage.setItem(SAVED_WIDGET_OPEN, open ? '1' : ''); }
    catch (error) { /* storage blocked: the window still opens, it just forgets */ }
  }

  function widgetWasOpen() {
    try { return localStorage.getItem(SAVED_WIDGET_OPEN) === '1'; }
    catch (error) { return false; }
  }

  const rootPx = () =>
    parseFloat(getComputedStyle(document.documentElement).fontSize) || 16;

  // Clamped so the window can neither collapse to a strip nor cover the chrome:
  // the bar and the rail are both fixed, and a helper sitting on top of them
  // would hide what is running to say something about it.
  function widgetLimits() {
    const rem = rootPx();
    const style = getComputedStyle(document.documentElement);
    const px = (name) => {
      const value = style.getPropertyValue(name).trim();
      return value.endsWith('rem') ? parseFloat(value) * rem : parseFloat(value) || 0;
    };
    return {
      minW: 18 * rem, minH: 14 * rem,
      maxW: window.innerWidth - 2 * rem,
      maxH: window.innerHeight - px('--bar-h') - px('--rail-h') - 2 * rem,
    };
  }

  function widgetResize(width, height) {
    const win = $('widget-window');
    const limit = widgetLimits();
    const size = {
      w: Math.round(Math.max(limit.minW, Math.min(width, limit.maxW))),
      h: Math.round(Math.max(limit.minH, Math.min(height, limit.maxH))),
    };
    win.style.width = `${size.w}px`;
    win.style.height = `${size.h}px`;
    // The class is what lets the log take the slack; see widget.css.
    win.classList.add('widget-sized');
    return size;
  }

  for (const grip of document.querySelectorAll('.widget-grip')) {
    const axis = grip.dataset.grip;
    // Pointer capture, so a drag that leaves a six-pixel handle keeps going —
    // without it the resize stops the moment the cursor outruns the edge, which
    // it does immediately.
    grip.addEventListener('pointerdown', (event) => {
      const box = $('widget-window').getBoundingClientRect();
      const from = { x: event.clientX, y: event.clientY, w: box.width, h: box.height };
      grip.setPointerCapture(event.pointerId);
      event.preventDefault();
      const move = (moved) => remember(SAVED_WIDGET_SIZE, widgetResize(
        axis === 'top' ? from.w : from.w + (from.x - moved.clientX),
        axis === 'left' ? from.h : from.h + (from.y - moved.clientY)));
      const done = () => {
        grip.removeEventListener('pointermove', move);
        grip.removeEventListener('pointerup', done);
        grip.removeEventListener('pointercancel', done);
      };
      grip.addEventListener('pointermove', move);
      grip.addEventListener('pointerup', done);
      grip.addEventListener('pointercancel', done);
    });
    // Each edge answers only its own axis, which is what a separator with an
    // orientation promises. The corner takes no focus: it is the two edges at
    // once, and there is nothing it can do that they cannot.
    grip.addEventListener('keydown', (event) => {
      const step = 2 * rootPx();
      const by = axis === 'top'
        ? { ArrowUp: [0, step], ArrowDown: [0, -step] }[event.key]
        : { ArrowLeft: [step, 0], ArrowRight: [-step, 0] }[event.key];
      if (!by) return;
      event.preventDefault();
      const box = $('widget-window').getBoundingClientRect();
      remember(SAVED_WIDGET_SIZE,
        widgetResize(box.width + by[0], box.height + by[1]));
    });
  }

  // Reapplied on load, and re-clamped when the viewport changes: a size that fit
  // yesterday's window can cover today's chrome, and the preference is worth
  // keeping through that rather than forgetting.
  function widgetRestoreSize() {
    const kept = saved(SAVED_WIDGET_SIZE);
    if (kept && kept.w && kept.h) widgetResize(kept.w, kept.h);
  }
  widgetRestoreSize();
  window.addEventListener('resize', widgetRestoreSize);

  // The only way another script may reach the widget. Kept deliberately small:
  // a page tells the widget something, and the widget decides how to show it.
  window.Widget = {
    say: widgetSay,
    note: widgetNote,
    offer: (text) => { WIDGET_RUN_ASK = text; widgetOffer(); },
    about: widgetAbout,
  };

  // A widget left open and then the page reloaded is the case this feature is
  // for: the reader never touched Close, so nothing here should look closed
  // either — the log restored from the lab, on this open exactly as on any other.
  if (widgetWasOpen()) {
    widgetSetOpen(true);
    widgetDraw();
  }
})();
