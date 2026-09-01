// The lab's LLM widget, on every surface. One definition: this file builds its
// own launcher and window into <body> rather than three pages carrying the same
// thirty lines of markup — the way lab.js already builds the footer rail and the
// theme control. Loads after lab.js, whose escapeHtml it uses.
//
// One route behind it (POST /api/widget/stream, which sends the answer as it
// is written) plus the GET that serves the model list and starters and the two
// that read and reset the conversation. Nothing here writes a run, a ledger row or a number, which is
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
      <button id="widget-new" class="widget-close" type="button" aria-label="New chat" title="Start a new conversation about this experiment">↻</button>
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

  // The generation as of right now, for a caller that is not itself starting
  // a draw and so has no `mine` of its own to be handed — `widgetAsk` reads
  // this before it posts, then asks `supersedes` the same question every
  // draw asks: did a newer one start while I was waiting.
  function currentGeneration() {
    return DRAW_SEQ;
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

  // What has become of the log by the time an answer to a question comes back.
  // Three things can be true of it, and only one of them means "paint here":
  //
  //   'gone'  — the reader has left the thread they asked in. The answer is
  //             about a conversation that is no longer on screen, and drawing
  //             it under the header that is would be the misattribution this
  //             whole task exists to rule out. Nothing is lost by refusing:
  //             the lab wrote the exchange to `intended`'s own history the
  //             moment it answered, so the next draw of that thread shows it.
  //   'stale' — still the same thread, so nothing could be misattributed, but
  //             a draw this answer is not part of is about to repaint the log
  //             (or already has). Painting into it would either be wiped a
  //             frame later or, worse, be wiped along with the question,
  //             because that draw's history GET was very likely issued before
  //             this answer had been written and so carries neither. This is
  //             the case that must not be a silent drop: the reader would be
  //             left with no question, no answer and no explanation, which is
  //             the same silent absence this file refuses for a note.
  //   'here'  — nothing has moved; say it where it was asked.
  //
  // `drawIntervened` is handed in rather than read here because whether a draw
  // has settled is state the DOM half owns; everything else this decides is
  // arithmetic on `DRAW_SEQ` and a string compare, which is what keeps the
  // decision itself testable without a page. It is a separate signal from
  // `supersedes` and both are needed: a question asked while the draw that
  // opened the widget is still in flight carries *that* draw's generation, so
  // nothing supersedes it, and yet that same draw is about to clear the log.
  //
  // It is named for what the caller must have established — that a draw stood
  // between the question and its answer — and not for `drawPending`, the flag
  // it is usually computed from, because reading that flag *here* (that is,
  // after the await) is precisely the bug this argument exists to rule out.
  // `drawPending` is true only while the newest draw is still running, and the
  // normal ordering is that it stops running long before an answer comes back:
  // a history GET is milliseconds and a model turn is seconds. A caller that
  // only looked at the flag on the way out would therefore find it false in
  // exactly the case it was meant to catch — the draw that erased the reader's
  // question having already finished doing so. So the caller captures the flag
  // synchronously before it posts and ORs that reading with the one it takes
  // afterwards, and hands the answer in; `widgetAsk` is the only caller, and
  // `widget_thread.test.js` pins that it still does it that way round.
  function replyFate(mine, intended, drawIntervened) {
    if (!stillCurrent(intended)) return 'gone';
    if (drawIntervened || supersedes(mine)) return 'stale';
    return 'here';
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
    // New Chat's own title has to track the same fact for the same reason:
    // there is no experiment to end a conversation "about" on the general
    // thread, and a title that said so anyway would be exactly the kind of
    // inaccurate label this file refuses everywhere else a control describes
    // which thread it acts on.
    $('widget-new').title = general
      ? 'Start a new conversation'
      : 'Start a new conversation about this experiment';
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

  // Every kind of line this log can hold, and the whole list of them:
  //
  //   you      the reader's question
  //   bot      the model's reply
  //   tool     a tool the model called on the way to one — the log's record of
  //            which real records an answer stands on. Not `note`: the lab
  //            speaking about itself is a different voice from the model
  //            reaching for something, and a reader has to be able to tell them
  //            apart at a glance without reading either.
  //   meta     the token account under a reply, a bill and never a measurement
  //   note     the lab saying what it just did
  //   err      what went wrong
  //
  // `thinking` is deliberately absent: it is built by `widgetThinking`, lives
  // for the length of one wait and is never said. Nothing that passes through
  // here is ephemeral.
  const WIDGET_KINDS = ['you', 'bot', 'tool', 'meta', 'note', 'err'];

  function widgetSay(kind, text) {
    const log = $('widget-log');
    // The empty state goes on the first thing said, not on the first reply: a
    // panel of examples sitting above your own question reads as a menu you
    // failed to use. A `note` is not a turn in the conversation — it is the lab
    // saying what it just did, and it can arrive at a widget the reader has never
    // opened — so it lands under the examples instead of clearing them.
    const offer = log.querySelector('.widget-empty');
    if (offer && kind !== 'note') offer.remove();
    // A kind this page does not know renders as a bare line and nothing else.
    // The class is written straight into the markup, so an unrecognised one is
    // both a styling question and — since the log is fed by a route — an
    // injection one; the answer to both is the same list. Inert rather than
    // dropped: a lab that grew a seventh kind must not make the reader's own
    // history disappear on the way to a page that has not been updated yet,
    // and the text is the part that carries what was said.
    const known = WIDGET_KINDS.includes(kind) ? ` ${kind}` : '';
    log.insertAdjacentHTML('beforeend',
      `<div class="widget-msg${known}">${escapeHtml(text)}</div>`);
    log.scrollTop = log.scrollHeight;
  }

  // --- the answer, as it is written ------------------------------------------
  // The reply used to land in one piece, one round trip after Send: a stalled
  // widget, then a wall of text. `/api/widget/stream` sends the same turn as
  // server-sent events — the pieces as the model writes them, then one final
  // event carrying the reply the lab now holds and its token account.
  //
  // Three rules the rest of this file already lives by, restated for a reply
  // that arrives over time rather than all at once:
  //
  //   * The final event has the last word. The pieces are how the answer
  //     arrived; the reply is read back out of the conversation log the lab
  //     just wrote, and the bubble adopts it (`widgetFinish`). A page that
  //     kept the concatenated pieces would be a second, private account of a
  //     turn the lab is the only copy of — the one thing this file refuses
  //     everywhere else.
  //   * A fragment is never left looking finished. A stream that dies keeps
  //     what did arrive — the reader watched it, pretending otherwise is its
  //     own lie — but marked as stopped, with the error line under it. An
  //     empty bubble is removed instead: it has nothing to say.
  //   * Every piece re-asks whether it still belongs on this screen. A draw
  //     can clear the log mid-answer, and the reader can leave the thread
  //     while it is still being written; both stop the typing where it is and
  //     leave the fate check at the end of `widgetAsk` to decide honestly
  //     between dropping the answer and redrawing the thread from the lab.

  function widgetLiveReply() {
    // Built through widgetSay so the empty state clears exactly as it does for
    // a reply that arrives whole, then filled with `textContent` — assigning
    // text, never markup, which is the same guarantee escapeHtml gives the
    // other writers.
    widgetSay('bot', '');
    const el = $('widget-log').lastElementChild;
    el.classList.add('streaming');
    return el;
  }

  function widgetType(el, text) {
    const log = $('widget-log');
    el.textContent += text;
    log.scrollTop = log.scrollHeight;
  }

  function widgetFinish(el, reply) {
    const log = $('widget-log');
    if (!el || !log.contains(el)) { widgetSay('bot', reply); return; }
    el.classList.remove('streaming');
    el.textContent = reply;
    log.scrollTop = log.scrollHeight;
  }

  // Two lines, out of the three statuses this page can be sent — and a stated
  // gap, because the one it most wants is one it can no longer be told.
  //
  // The third status is `not_filed`: the tool-hop guard stopped the run, so
  // the reply above is the widget refusing rather than an answer, and nothing
  // about it is going to be kept. It gets no line for the same reason a
  // refused question gets none — the refusal is the answer the reader is
  // already reading, and a second line under it would only say so again.
  //
  // The decision about keeping a turn outlives the response: both answer paths
  // file after they answer (`backends._defer_memory`), so what reaches this
  // reader is the status at the moment the last event was written — the
  // decision pending, or no judge to make it. The verdict itself lands in
  // widget.db afterwards, on a thread nothing on this page is attached to, and
  // nothing asks for it again. So a reader is told a decision is coming and is
  // never told it succeeded. The `saved` and `not_saved` lines were deleted
  // rather than left here for an event that no longer arrives: closing the gap
  // means carrying the turn id out to this page and reading
  // `widget_turn_log.memory_update_id` back over a route that does not exist
  // yet, which is a feature and not a line of copy.
  //
  // A refused question has no line either, and needs none: its status rides on
  // the same event as the reply (`backends._refused`), which `widgetStream`
  // does not route here — and the refusal is the answer the reader is already
  // reading.
  function widgetMemoryStatus(memory) {
    const copy = {
      // Not one line for both, deliberately: nobody judged the unavailable
      // turn, so saying it was off-topic — or that a decision is still coming
      // — would be putting a verdict in a reader's mouth that no model gave.
      unavailable: 'No memory saved: the memory helper could not be reached.',
      // The answer comes out before the decision about keeping it is taken, so
      // this is what is true when the turn lands: not a save, not a refusal,
      // and not silence — which would read as the lab finding nothing worth
      // keeping.
      pending: 'Nothing filed yet: still deciding whether to keep this turn.',
    };
    return copy[memory && memory.status] || '';
  }

  function widgetStopped(el) {
    const log = $('widget-log');
    if (!el || !log.contains(el)) return;
    el.classList.remove('streaming');
    // Nothing arrived, so there is nothing to keep: an empty bubble above an
    // error line reads as an answer of no words rather than as no answer.
    if (!el.textContent) { el.remove(); return; }
    el.classList.add('stopped');
  }

  // The stretch between Send and the first piece used to be the one part of a
  // turn the log said nothing about: the reader's own bubble, then nothing,
  // whether the lab was composing or off calling a tool. This line names that
  // wait — "Thinking…", swapped for the tool's name when a status event says
  // one is running. It is deliberately not a turn: never written through
  // `widgetSay` (whose empty-state handling is for things said), never part
  // of any history the lab keeps — a redraw of this thread comes back without
  // it because it was never in it — and gone the moment the answer starts or
  // the turn ends, whichever comes first. `textContent`, never markup: the
  // swap writes a name the stream sent, untrusted like every other string
  // that arrives over a wire.
  function widgetThinking() {
    const log = $('widget-log');
    const el = document.createElement('div');
    el.className = 'widget-msg thinking';
    el.textContent = 'Thinking…';
    log.append(el);
    log.scrollTop = log.scrollHeight;
    return el;
  }

  // Removal, guarded by the same `contains` check every other writer to this
  // log makes before touching it: a draw may already have cleared the log,
  // and a line the log no longer holds is not this turn's to act on — the
  // screen has moved on, and the fate check at the end of `widgetAsk` is
  // what decides honestly where the turn itself lands.
  function widgetThinkingOver(el) {
    const log = $('widget-log');
    if (el && log.contains(el)) el.remove();
  }

  // One SSE reader: `data: ` lines carrying one JSON object each, events
  // separated by a blank line. Deltas go to `onDelta` as they land, status
  // events — the lab naming the tool it is calling — to `onStatus`; the final
  // event is returned. An `error` event is thrown, because that is what it is —
  // the stream's own way of saying the answer never finished, once the status
  // code has been spent on the first piece.
  async function widgetStream(path, body, onDelta, onStatus, onMemoryStatus) {
    const res = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      // Keep the request alive while the reader navigates between surfaces.
      keepalive: true,
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(data.detail || res.statusText);
    }
    let buffer = '';
    let final = null;
    // `flush` is the difference between "the connection is still open, so the
    // tail of the buffer is half an event" and "it closed, so the tail is all
    // there will ever be". Splitting without that distinction would parse a
    // half-arrived event and throw on it.
    const drain = (flush) => {
      const parts = buffer.split('\n\n');
      buffer = flush ? '' : parts.pop();
      for (const part of parts) {
        const line = part.split('\n').find((l) => l.startsWith('data: '));
        if (!line) continue;
        let event;
        try {
          event = JSON.parse(line.slice(6));
        } catch (error) {
          throw new Error('the lab sent an answer this page could not read');
        }
        if (event.error) throw new Error(event.error);
        // A status event is the lab saying what it is doing right now — a
        // tool being called — not part of the answer. It is ephemeral by
        // contract (no history ever holds one), so it must not fall through
        // to the `final = event` line below the way everything without a
        // `delta` otherwise does: a stream that died after a status event
        // would then hand back the chatter as the reply the lab supposedly
        // holds, which is exactly the substitution this file refuses.
        if (event.status != null) { onStatus(event.status); continue; }
        if (event.memory != null && event.reply == null) {
          if (onMemoryStatus) onMemoryStatus(event.memory);
          continue;
        }
        if (event.delta != null) onDelta(event.delta);
        else if (event.reply != null) final = event;
      }
    };
    if (res.body) {
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      for (;;) {
        const step = await reader.read();
        if (step.value) buffer += decoder.decode(step.value, { stream: true });
        drain(step.done);
        if (step.done) break;
      }
    } else {
      // No streaming body to read (an old browser, a harness that buffers):
      // the same events, parsed the same way, all at once. The answer is not
      // lost — it simply does not type itself out.
      buffer = await res.text();
      drain(true);
    }
    if (!final) {
      throw new Error('the answer stopped before the lab said what it holds');
    }
    return final;
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

  // One lock, for every control that can trigger `widgetAsk`'s POST or
  // `widget-new`'s DELETE — the two backend calls that race each other on
  // the same `thread_id` with no coordination on the lab's side (see the
  // comments on `widgetAsk` and on the `widget-new` click handler). A
  // starter chip is a third way to reach `widgetAsk`, not a decoration on
  // top of it — `widgetLog`'s delegated click listener below calls it
  // exactly the way the form's submit handler does — so a chip left live
  // while `widget-send`/`widget-new` are disabled would be the one door
  // this lock left open: clickable, and silently starting the very request
  // the lock exists to keep from overlapping another. Disabling it outright,
  // rather than merely leaving it clickable-but-ignored, is the honest
  // choice `widgetOffer` already makes about the model list being missing —
  // a live-looking control that does nothing on a click is its own small
  // lie about what the reader can do right now.
  //
  // `widgetLocked` is read by `widgetOffer`'s own chip template, not just
  // applied after the fact, because a chip can be *born* during the locked
  // window — a run landing mid-delete or mid-ask calls `Widget.offer`,
  // which rebuilds the chips from scratch — and a chip rendered live and
  // then left unpatched would be exactly the gap this exists to close.
  let widgetLocked = false;
  function widgetLock(locked) {
    widgetLocked = locked;
    $('widget-send').disabled = locked;
    $('widget-new').disabled = locked;
    for (const chip of document.querySelectorAll('.widget-starter')) chip.disabled = locked;
  }

  // Rendered only while the log is empty. Called again when a run lands, so a
  // widget left open and untouched picks up the chip for the run just finished.
  function widgetOffer() {
    const log = $('widget-log');
    // A note is not a conversation, so a log holding only notes is still a log
    // nobody has asked anything in — and the reader whose widget was opened *by*
    // a note is exactly the reader who has not seen the examples yet.
    if (log.querySelector('.widget-msg:not(.note)')) return;
    // A chip is an invitation to ask, and while the model list is missing
    // `widgetAsk` refuses to ask at all — so offering one would be advertising
    // a button that answers a click with a refusal. It would not even survive
    // being offered: `widgetDrawThread` says the options error immediately
    // after this, and `widgetSay` clears the empty state for every kind but a
    // note, so on an empty thread the chips would be inserted and removed in
    // the same pass. Writing what you are about to erase is the shape of bug
    // this file has spent several rounds removing; the honest version is not
    // to write it.
    if (widgetOptionsError) return;
    if (!WIDGET_STARTERS.length && !WIDGET_RUN_ASK) return;
    const chip = (text, extra = '') =>
      `<button type="button" class="widget-starter${extra}"${widgetLocked ? ' disabled' : ''}>`
      + `${escapeHtml(text)}</button>`;
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
    widgetSayAfterDraw('note', text, intended);
  }

  // The waiting half of the paragraph above, on its own because a note is not
  // the only line the lab keeps no copy of: `widgetAsk` says an `err` under
  // exactly the same conditions — something worth telling the reader, written
  // here and nowhere else, arriving while a redraw that would wipe it is
  // still in flight. Both must wait for that draw rather than race it, and
  // both must then re-ask whether they still belong on the screen they are
  // about to write to. One implementation, so the two cannot drift apart on
  // which of those two checks matters.
  function widgetSayAfterDraw(kind, text, intended) {
    const mine = drawnGeneration;
    widgetDrawing.catch(() => {}).then(() => {
      if (supersedes(mine) || !stillCurrent(intended)) return;
      widgetSay(kind, text);
    });
  }

  // Delegated, because the offer is rebuilt whenever a run lands. No lock
  // check belongs here: a chip is a real `<button>`, `widgetLock` sets its
  // `disabled` attribute exactly like the other two controls, and a
  // disabled button never dispatches a click for a delegated ancestor
  // listener to see in the first place — the same native behaviour already
  // relied on for `widget-send`/`widget-new` themselves. Reaching this
  // handler at all is proof the chip clicked was not locked.
  $('widget-log').addEventListener('click', (event) => {
    const starter = event.target.closest('.widget-starter');
    if (!starter) return;
    widgetAsk(starter.textContent);
  });

  // This file's one log writer that used to carry no guard at all: it reads
  // `widgetThread()` at post time and, after the round trip, said `bot` and
  // `err` unconditionally. That was fine while every surface shared one
  // thread — there was nowhere else for the reply to belong. Once the header
  // started naming a thread, an answer landing here after the reader left it
  // (clicked the header, opened a different experiment, heard about one from
  // another tab) became exactly the lie this task exists to rule out: an
  // answer about A rendered under B's header, with nothing on the row saying
  // so. `intended`/`mine` are captured before the post, the same shape as
  // `widgetNote`'s, and checked again after.
  //
  // The exchange is not lost when the reader has left the thread — the lab
  // already wrote it to `intended`'s own history the moment it answered, and
  // whoever looks at `intended` next sees it there, drawn fresh, the way any
  // earlier turn is. What must not happen is drawing it here, now, under a
  // header that did not produce it, so that render is dropped rather than
  // misattributed.
  //
  // Dropping was, for one round, what happened in the other superseded case
  // too — a redraw of the *same* thread starting while the question was in
  // flight (a board open on the experiment already open, a cross-tab
  // `storage` echo of the same id, the launcher shut and reopened). There the
  // header is right and nothing could be misattributed, so a drop bought no
  // honesty and cost the reader their own turn: that redraw's history GET
  // races the still-pending POST and usually wins, so the question goes with
  // the answer and the reader is left with no trace of either. `replyFate`
  // separates the two: `'gone'` still refuses, `'stale'` redraws. The redraw
  // is the honest way out precisely because the lab is the only copy of the
  // transcript — its history now holds this exchange, a draw started after
  // the POST resolved is guaranteed to read it back, and a draw rebuilds the
  // log from nothing, so it cannot duplicate a turn the way a second
  // `widgetSay` could. It also takes a newer generation than whatever draw
  // overtook us, so that one stands down rather than repainting over it.
  //
  // The redraw carries the token account too, not just the answer:
  // `conversation_memory` puts `usage_metadata` on the stored `AIMessage` and
  // carries it through the checkpointer, so the history this redraw fetches
  // already has the right numbers sitting on the right turn, and
  // `widgetDrawThread`'s own loop renders them the same way it renders every
  // other redrawn reply's meta line. Nothing here has to park a count under
  // "whatever turn happens to be last" the way an appended-after-the-fact
  // number would have to — the account is drawn from the same row the answer
  // is, so it cannot land on the wrong one. (This used to be the one thing a
  // `'stale'` redraw could not show; it changed when the account moved onto
  // the turn itself, in the log the lab keeps, rather than living only in the
  // reply this function got back.)
  //
  // A failed POST takes neither route. There is nothing new in history to
  // redraw — that is what failed — and the error line exists nowhere but
  // here, so it is handed to `widgetSayAfterDraw` to be said once the draw
  // that would have wiped it has finished, the same way a note is.
  //
  // Asking and ending a conversation are mutually exclusive in meaning — you
  // cannot coherently be asking a question and ending the thread it belongs
  // to at the same instant — so `widgetLock(true)` holds `widget-new` down
  // for the same stretch `widget-send` already is, and every starter chip
  // with it: a chip is a second way to reach this very function (see the
  // delegated click listener on `widget-log`), not a decoration outside the
  // lock. Without this, `forget()` on the New Chat side calls
  // `saver().delete_thread(name)` with no coordination against the
  // checkpoint write this function's own `agent.stream` (inside
  // `widgetStream('/api/widget/stream', ...)`, on the lab's side) makes to the
  // very same `thread_id`. If that write lands after the delete, the thread New Chat
  // just ended regrows exactly one turn — this function's own `'stale'`
  // branch would still redraw honestly from whatever the backend now holds,
  // so the screen never lies, but the substantive promise ("this
  // conversation has ended") would be broken regardless. Locking every one
  // of those controls before either request leaves this tick is what keeps
  // the two operations from ever being in flight together on any path a
  // reader can actually reach — a slower click, not a second open tab.
  async function widgetAsk(message) {
    const intended = widgetThread();
    const mine = currentGeneration();
    // Captured here, with the other two, and for the same reason: every
    // writer to this log takes what it needs to judge itself by *before* it
    // awaits anything. A draw already in flight when Send is pressed — the
    // one New Chat starts, the one the launcher starts, the one a `storage`
    // echo from another tab starts — will clear the log, and it will almost
    // certainly have done so and finished by the time this answer lands, at
    // which point `drawPending` reads false and says nothing happened. What
    // did happen is that the reader's own question was wiped. This reading
    // and the one taken after the await are ORed together below, so either
    // end of the wait is enough to make the answer a `'stale'` one.
    const wasPending = drawPending;
    widgetSay('you', message);
    const model = $('widget-model').value;
    if (!model) {
      // Posting with `model: ''` does not fail — `ask()` on the lab's side
      // quietly substitutes its own default, and its reply carries no field
      // naming which model actually ran, so there is no way to say so
      // honestly afterwards. An empty selection here — the model list
      // failed to load, most likely — is refused before the fact instead,
      // which is the only way the dropdown is never shown empty while a
      // real model answers anyway.
      widgetSay('err', 'No model is available, so this question was not '
        + 'sent — the model list failed to load. Reopen the helper to retry.');
      return;
    }
    widgetLock(true);
    // The wait is named up front, where the reply's bubble deliberately is
    // not: "Thinking…" is a claim about this page — a question is out and
    // nothing has come back — which is true from this line on, where an
    // empty bubble would be a claim about an answer the lab has not started
    // making. It lasts exactly as long as the wait does: the first piece
    // removes it, and the catch and finally below clear it on every other
    // way out, so it can never outlive the turn it announces.
    const thinking = widgetThinking();
    // The bubble the answer is typed into, created on the first piece rather
    // than up front: a bubble that appears and then never fills is a claim
    // the lab has not made yet.
    let live = null;
    try {
      const data = await widgetStream('/api/widget/stream',
        { message, model, thread: intended },
        (delta) => {
          // Every piece re-asks the two questions the answer as a whole is
          // asked below. A newer draw, or a reader who has left the thread,
          // stops the typing here — the log has moved on, and the fate check
          // at the end is what decides whether this turn is dropped or the
          // thread redrawn from the lab. `contains` is the same fact read a
          // second, cheaper way: a bubble the log no longer holds was wiped
          // by a draw, whatever the generation says.
          if (supersedes(mine) || !stillCurrent(intended)) { live = null; return; }
          // The first piece is the answer starting, which is the end of the
          // wait the indicator names — it goes here, where the bubble that
          // replaces it is born, so the two never sit on screen together.
          widgetThinkingOver(thinking);
          if (live && !$('widget-log').contains(live)) live = null;
          if (!live) live = widgetLiveReply();
          widgetType(live, delta);
        },
        (status) => {
          // The swap answers to the same rule the live bubble does: only a
          // line the log still holds is this turn's to retitle. A redraw
          // that wiped it means the log has moved on, and recreating it
          // would paint a claim about this turn onto a screen that is no
          // longer showing it — so a wiped indicator stays gone. The name
          // is assigned as text, never markup, because it arrived over the
          // stream like every other untrusted string.
          if ($('widget-log').contains(thinking)) {
            thinking.textContent = `calling ${status}…`;
          }
        },
        (memoryStatus) => {
          const fate = replyFate(mine, intended, wasPending || drawPending);
          if (fate !== 'here') return;
          const copy = widgetMemoryStatus(memoryStatus);
          if (copy) widgetSay('meta', copy);
        });
      const fate = replyFate(mine, intended, wasPending || drawPending);
      if (fate === 'gone') return;
      if (fate === 'stale') { widgetDraw(); return; }
      // The lab's own reading of what it just wrote replaces what was typed:
      // the pieces were how the answer arrived, this is the turn the log
      // holds, and the two must not be allowed to differ on screen.
      widgetFinish(live, data.reply);
      // The token account, when the backend reported one — an unreported
      // account renders nothing rather than a made-up zero.
      if (data.input_tokens != null) {
        widgetSay('meta', `out ${data.output_tokens} in ${data.input_tokens} tok.`);
      }
    } catch (error) {
      // The wait line goes before the error is said: an error line landing
      // under a still-pulsing "Thinking…" would read as one more thing
      // being worked on, when the truth is that the work just stopped.
      widgetThinkingOver(thinking);
      // Whatever had arrived stays, marked as stopped rather than dressed up
      // as a finished reply; an empty bubble goes, having said nothing.
      widgetStopped(live);
      const fate = replyFate(mine, intended, wasPending || drawPending);
      if (fate === 'gone') return;
      if (fate === 'stale') { widgetSayAfterDraw('err', error.message, intended); return; }
      widgetSay('err', error.message);
    } finally {
      // Cleared once more on every way out, the success paths included: a
      // turn whose answer arrived whole (a final event with no pieces)
      // never reached the first-delta removal, and an indicator that
      // outlived its turn would be this page claiming the lab is thinking
      // about nothing.
      widgetThinkingOver(thinking);
      widgetLock(false);
      $('widget-input').focus();
    }
  }

  // The model list and the starters are served, not kept here — fetched once, on
  // the first open. Returns its promise rather than swallowing it, so a caller
  // that needs the options in place first — `widgetDrawThread` needs
  // `WIDGET_STARTERS` before it can decide whether to offer them — can wait
  // on it with `.then` instead of racing it.
  let widgetOptionsLoaded = false;
  // Set on a failed load, cleared on a successful one — read by
  // `widgetDrawThread`, not said here, because every caller of this function
  // is immediately followed by a draw (see `widgetDraw`) whose own
  // `log.innerHTML = ''` would silently erase an error said at this point
  // the instant it repaints. An error about the model list disappearing
  // under the very next redraw is exactly the kind of loss this file no
  // longer allows itself anywhere else in this task.
  let widgetOptionsError = null;
  async function widgetLoadOptions() {
    if (widgetOptionsLoaded) return;
    try {
      const data = await api('/api/widget');
      $('widget-model').innerHTML = data.models.map((m) =>
        `<option value="${escapeHtml(m.value)}"${m.value === data.default ? ' selected' : ''}>`
        + `${escapeHtml(m.label)}</option>`).join('');
      WIDGET_STARTERS = data.starters || [];
      widgetOptionsLoaded = true;
      widgetOptionsError = null;
    } catch (error) {
      widgetOptionsError = error.message;
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
    // silent absence is most dangerous). An `if`/`else` rather than an early
    // return, because the options-list error below must still be said either
    // way — it is a separate fact about this draw, not a reason to skip it.
    if (Array.isArray(read.turns)) {
      for (const turn of read.turns) {
        widgetSay(turn.role, turn.text);
        // The account travels on the turn itself now (`conversation_memory`
        // carries `usage_metadata` through the checkpointer), so a redrawn
        // reply gets the same meta line the live one showed — checked with
        // `!= null` rather than truthiness so a real, reported zero still
        // renders instead of being read as "nothing reported".
        if (turn.input_tokens != null) {
          widgetSay('meta', `out ${turn.output_tokens} in ${turn.input_tokens} tok.`);
        }
      }
      if (!read.turns.length) widgetOffer();
    } else {
      widgetSay('err', 'The lab answered, but this thread’s history came '
        + 'back in a shape this page could not read.');
    }
    // Said here, once the paint above is done, rather than inside
    // `widgetLoadOptions` itself — see the comment there. `widgetAsk`
    // separately refuses to post with no model selected, so this line is
    // the explanation for why the dropdown is empty and starters are
    // missing, not the only thing standing between a reader and a silent
    // substitution.
    if (widgetOptionsError) {
      widgetSay('err', `Could not load the model list (${widgetOptionsError}). `
        + 'Asking is refused until it does; reopening the helper tries again.');
    }
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
  //
  // `drawPending` says whether the newest draw has finished, which is the one
  // thing a generation number cannot say on its own: a question asked while
  // the draw that opened the widget is still in flight carries that draw's
  // own generation, so `supersedes` is false about it, and yet it is about to
  // clear the log out from under the answer. Only the newest draw's
  // settlement clears the flag — an older one finishing late says nothing
  // about whether the screen has stopped moving, so it leaves the flag alone.
  let widgetDrawing = Promise.resolve();
  let drawnFor = null;
  let drawnGeneration = 0;
  let drawPending = false;
  function widgetDraw() {
    drawnFor = widgetThread();
    drawnGeneration = nextGeneration();
    const mine = drawnGeneration;
    drawPending = true;
    // `finally`, not `then`: neither half of the chain is allowed to reject
    // and both take care not to, but a flag that stayed true forever because
    // one of them did anyway would quietly turn every later answer into a
    // redraw — and `finally` also passes a rejection on untouched, leaving
    // this promise exactly as rejectable (that is, not) as it was before.
    widgetDrawing = widgetLoadOptions().then(() => widgetDrawThread(mine))
      .finally(() => { if (!supersedes(mine)) drawPending = false; });
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

  // New Chat: the only control that ends a conversation, and it ends exactly
  // the one on screen. Ending a conversation means asking the lab to forget
  // it, not merely blanking the log — a cleared screen over a conversation
  // widget.db still holds would be this file lying about what the model
  // remembers, the same rule a note or a reply already answers to. `intended`
  // is captured before the DELETE, the same discipline every other writer to
  // this log follows: the reader is free to open a different experiment
  // while the request is in flight, and this button must end the thread it
  // was pressed on, never whatever the screen has since moved to.
  //
  // A successful forget redraws only when the reader is still on that
  // thread. Leaving it already started its own draw of wherever they went —
  // `widgetAbout` sees to that — so forcing a second draw here would be a
  // redundant fetch of a screen nobody is looking at, the same reasoning
  // `replyFate`'s 'gone' case already applies to a reply that arrives after
  // the reader has left. A failed forget is said through
  // `widgetSayAfterDraw`, not `widgetSay` directly, for the reason a note
  // already is: a draw already in flight for this thread — the widget's own
  // opening draw, another tab's switch echoed back through `storage` — would
  // otherwise clear the log a moment after the error line lands in it, and
  // `widgetSayAfterDraw` is the one place that wait already lives, rather
  // than a second copy of it invented here.
  $('widget-new').addEventListener('click', async () => {
    const intended = widgetThread();
    // Send and every starter chip are locked down for the same stretch, and
    // for the same reason `widgetAsk` locks New Chat down during its own
    // request: the DELETE this triggers races `widgetAsk`'s own checkpoint
    // write for this thread with no coordination on the lab's side, and a
    // question that slips in between this click and the DELETE landing —
    // typed, or a single click on a chip — could regrow the thread this
    // button just ended. A chip is worth naming here specifically: the
    // starters are only ever visible on an empty thread, which is exactly
    // the state this button produces, so the invitation to ask is sitting
    // right there under the button that just ended the conversation unless
    // it is locked too. See the comment on `widgetAsk` for the full shape
    // of that race — this is the other half of the same guard.
    widgetLock(true);
    try {
      await api('/api/widget/history?thread='
                + encodeURIComponent(intended), null, 'DELETE');
      if (stillCurrent(intended)) widgetDraw();
    } catch (error) {
      widgetSayAfterDraw('err', error.message, intended);
    } finally {
      widgetLock(false);
    }
  });

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
  // remembered under the same `raglab:` prefix as the settings and the last run.
  const SAVED_WIDGET_SIZE = 'raglab:widget-size';

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
  // `say` is deliberately not here: it is the raw log writer, with no thread,
  // generation or in-flight-draw check, so a caller on any surface could have
  // written a line under any header at any moment, bypassing every guard this
  // file keeps. Nothing outside this file ever called it — `note`, `about`
  // and `offer` are the whole of what a page has ever said to the widget —
  // so the door it opened is closed rather than left standing for the next
  // caller to find.
  window.Widget = {
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
