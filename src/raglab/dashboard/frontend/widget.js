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
    <span>Lab helper</span>
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

  mountWidget();

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
  // *lab's* last run and not the last conversation: widget memory is an
  // in-process checkpointer keyed to a page-scoped session id, so a reload
  // genuinely forgets, and a chip claiming otherwise would be a panel lying
  // about what produced it.
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
  function widgetNote(text) {
    const win = $('widget-window');
    if (win.hidden) {
      win.hidden = false;
      widgetLoadOptions();
    }
    widgetSay('note', text);
  }

  // Delegated, because the offer is rebuilt whenever a run lands.
  $('widget-log').addEventListener('click', (event) => {
    const starter = event.target.closest('.widget-starter');
    if (!starter) return;
    widgetAsk(starter.textContent);
  });

  // One conversation per page: the id is minted when the script loads and sent
  // with every ask, so a follow-up lands in the same thread and a reloaded page
  // starts clean — nothing persisted, nothing shared between tabs.
  const widgetSession = crypto.randomUUID();

  async function widgetAsk(message) {
    widgetSay('you', message);
    $('widget-send').disabled = true;
    try {
      const data = await api('/api/widget',
        { message, model: $('widget-model').value, session: widgetSession });
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
  // the first open.
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
      widgetOffer();
    } catch (error) {
      widgetSay('err', error.message);
    }
  }

  $('widget-launch').addEventListener('click', () => {
    const win = $('widget-window');
    win.hidden = !win.hidden;
    if (!win.hidden) { widgetLoadOptions(); $('widget-input').focus(); }
  });

  $('widget-settings').addEventListener('click', () => {
    const row = $('widget-config');
    row.hidden = !row.hidden;
  });

  $('widget-close').addEventListener('click', () => { $('widget-window').hidden = true; });

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
  };
})();
