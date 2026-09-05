// Shared by all three surfaces — the Laboratory, the Inspector and the
// Leaderboard — loaded before each page's own script, the way sorttable.js
// already is. Holds only what those scripts, written months apart, turned
// out to need identically.

// The stricter of the two copies this page used to carry separately: escapes
// `"` as well as `&<>`. In a text node `&quot;` renders as `"`, so nothing
// looks different there; in an attribute value it closes a latent injection.
// The split plan as the one line a person reads: stages joined by ` / `, a
// separator quoted the way JSON writes it, a label boundary as `role=user`,
// a drift stage's markers as `drift or "…"`, and a stage's `when` only when
// it differs from its kind's default. The same rendering
// `configuration/split_plan.text()` produces, so a plan reads identically on
// a knob page, a sweep candidate, the board and the Inspector.
const PLAN_DEFAULT_WHEN = Object.freeze(
  { part: 'always', label: 'always', drift: 'always', separator: 'over-budget' });

function planText(stages) {
  if (typeof stages === 'string') return stages;
  if (!Array.isArray(stages)) return '';
  const atom = (a) => ('text' in (a || {}) ? JSON.stringify(a.text)
    : `${(a || {}).label}=${(a || {}).value}`);
  return stages.map((stage) => {
    const kind = (stage || {}).kind || '';
    let words;
    if (kind === 'document' || kind === 'part') words = [kind];
    else if (kind === 'drift') {
      words = ['drift'];
      for (const marker of stage.markers || []) words.push('or', JSON.stringify(marker));
    } else {
      words = [];
      for (const a of stage.atoms || []) {
        if (words.length) words.push(stage.join || 'or');
        words.push(atom(a));
      }
      if (!words.length) words = [kind];
    }
    if (kind in PLAN_DEFAULT_WHEN && stage.when && stage.when !== PLAN_DEFAULT_WHEN[kind]) {
      words.push(stage.when);
    }
    return words.join(' ');
  }).join(' / ');
}

function escapeHtml(text) {
  return String(text === null || text === undefined ? '' : text)
    .replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

// --- Day and Night ---------------------------------------------------------
// Two themes and a third setting, Auto, which is the absence of a choice: with
// nothing stored the page follows the machine, through the guarded media query
// in tokens.css. So Auto is stored by storing nothing, and the whole of the
// state this holds is one attribute on <html> and one key in localStorage.
//
// The attribute is also set by a one-line inline script in each page's head,
// before the first paint. That copy exists because this file is a request: run
// only from here, at the foot of the page, and a reader who chose Night sees
// the other theme flash on every navigation between the three surfaces. The
// two copies read the same key, which is the only thing they must agree on.
//
// All three surfaces are one origin, so the choice travels between them: the
// Inspector reads the same key the Laboratory wrote. That is why the theme
// control appears identical on every page rather than three times over.
const THEME_KEY = 'raglab-theme';
// Light, dark, neither — in that order, so the control reads as a range with
// the "I have not decided" end where a reader expects to find it.
const THEMES = [['day', 'Day'], ['night', 'Night'], ['', 'Auto']];

// Storage throws rather than returning null in real configurations — Safari's
// private browsing, a browser set to block site data. The switch has to keep
// working there; it just stops outliving the tab.
function readTheme() {
  try {
    const stored = localStorage.getItem(THEME_KEY);
    return stored === 'day' || stored === 'night' ? stored : '';
  } catch (error) {
    return '';
  }
}

// Auto clears the attribute rather than writing a third name: the media query
// in tokens.css is guarded on `:not([data-theme="day"])` and knows only the two
// themes, so any other value would leave the page on Day for ever and stop it
// following a machine that changes its mind at sunset.
function applyTheme(theme) {
  if (theme) document.documentElement.dataset.theme = theme;
  else document.documentElement.removeAttribute('data-theme');
}

function writeTheme(theme) {
  applyTheme(theme);
  try {
    if (theme) localStorage.setItem(THEME_KEY, theme);
    else localStorage.removeItem(THEME_KEY);
  } catch (error) {
    /* Applied but not remembered, which beats refusing to switch. */
  }
}

// Built here rather than written into all three pages: the same markup copied
// into three files is how a shared control comes to differ between them, and
// this one has a checked state and a listener that have to stay together. The
// pages carry the empty container; this fills it. Radios rather than buttons,
// so the arrow keys walk the group and the selection is the browser's state
// instead of a class somebody has to remember to move.
function mountThemeControl(host) {
  if (!host) return;
  const current = readTheme();
  host.setAttribute('role', 'radiogroup');
  host.innerHTML = THEMES.map(([value, label]) => {
    const id = `theme-${value || 'auto'}`;
    return `<input type="radio" name="raglab-theme" id="${id}" value="${value}"`
      + `${value === current ? ' checked' : ''}>`
      + `<label for="${id}">${escapeHtml(label)}</label>`;
  }).join('');
  host.addEventListener('change', (event) => {
    if (event.target.name === 'raglab-theme') writeTheme(event.target.value);
  });
}

// --- how tall a row is -----------------------------------------------------
// The second thing a reader gets to decide about these surfaces, kept beside
// the first for the same reasons: one key, one origin, so the choice travels
// between the four pages, and one place that builds the control so four copies
// of it cannot drift.
//
// Compact is the default and is the absence of a stored value, exactly as Auto
// is for the theme — the sheet's own `:root` is the compact one, and the
// attribute only ever names the departure from it.
const DENSITY_KEY = 'raglab-density';
const DENSITIES = [['', 'Compact'], ['comfortable', 'Comfortable']];

function readDensity() {
  try {
    return localStorage.getItem(DENSITY_KEY) === 'comfortable'
      ? 'comfortable' : '';
  } catch (error) {
    return '';
  }
}

function writeDensity(density) {
  if (density) document.documentElement.dataset.density = density;
  else document.documentElement.removeAttribute('data-density');
  try {
    if (density) localStorage.setItem(DENSITY_KEY, density);
    else localStorage.removeItem(DENSITY_KEY);
  } catch (error) {
    /* Applied but not remembered, which beats refusing to switch. */
  }
}

function mountDensityControl(host) {
  if (!host) return;
  const current = readDensity();
  host.setAttribute('role', 'radiogroup');
  host.innerHTML = DENSITIES.map(([value, label]) => {
    const id = `density-${value || 'compact'}`;
    return `<input type="radio" name="raglab-density" id="${id}" value="${value}"`
      + `${value === current ? ' checked' : ''}>`
      + `<label for="${id}">${escapeHtml(label)}</label>`;
  }).join('');
  host.addEventListener('change', (event) => {
    if (event.target.name === 'raglab-density') writeDensity(event.target.value);
  });
}

document.addEventListener('DOMContentLoaded', () => {
  mountThemeControl(document.getElementById('theme-control'));
  // Applied before the control is built, so a page opened by a reader who
  // chose comfortable last week draws that way rather than switching under
  // them once the script reaches the radio.
  writeDensity(readDensity());
  mountDensityControl(document.getElementById('density-control'));
});


// --- the row above the column names ----------------------------------------
// A spanning header saying which block of columns you have scrolled into.
// Both grid surfaces build one, so it is built here: a column moved between
// groups moves the bar with it, and a group can never claim a span it does not
// have, because the runs are derived from the columns rather than declared
// beside them.
//
// `sorttable.js` and `filtertable.js` both read the LAST row of `<thead>` as
// the headings, which is the whole reason this row can exist without either of
// them knowing about it.
//
// `columns` are the page's own column objects: `group` names the block, and
// `freeze` — where a column has one — is carried onto the group cell, but only
// where the group IS that one column. A spanning cell cannot be pinned to an
// edge that most of its columns are not on.
function groupRow(columns, groups) {
  if (!columns.some((column) => column.group)) return '';
  const runs = [];
  for (const column of columns) {
    const last = runs[runs.length - 1];
    if (last && last.name === column.group) { last.span += 1; continue; }
    runs.push({ name: column.group, span: 1, freeze: column.freeze || '' });
  }
  return '<tr class="group-row">' + runs.map((run) => {
    const group = groups[run.name] || { label: run.name || '' };
    const named = ['why-term', run.span === 1 ? run.freeze : '']
      .filter(Boolean).join(' ');
    return `<th scope="colgroup" colspan="${run.span}" class="${named}"`
      + `${group.step ? ` data-step="${group.step}"` : ''}`
      + `${group.brief ? ` data-brief="${escapeHtml(group.brief)}"` : ''}>`
      + `${escapeHtml(group.label)}</th>`;
  }).join('') + '</tr>';
}

// Appended to every sortable heading's note, so what a click does is said once
// for all four tables rather than at each of their columns — and said in the
// box a keyboard can open, which is why `sorttable.js` leaves its own native
// tooltip off any heading carrying one of these.
const SORT_LINE = '\n• Click to sort · again to reverse · a third time for '
  + 'the order it was served in.';


// --- the rows, from the keyboard -------------------------------------------
// A scroll region that takes focus can already be scrolled with the arrows,
// which moves the viewport and nothing else: there is no such thing as "the
// row I am on", so there is nothing for Enter to open and nothing for a
// screen reader to be told about. On a 167-row corpus or an 84-row board that
// is the difference between a table you can read and a table you can only
// look at.
//
// So the arrows move a row instead of the viewport — which scrolls the region
// anyway, because focusing a row brings it into view. Up and Down step,
// Home and End go to the ends, and Enter presses whatever the row's first
// control is: the open arrow on the board, the parts button on a corpus. The
// row itself is what takes focus, so the browser announces the whole row.
//
// Deliberately not a `role="grid"`: that is a contract about cell-by-cell
// navigation, and what these tables want is row-by-row. A table that says it
// is a grid and then does not behave like one is worse than a table.
function mountRowKeys(region) {
  const table = region && region.querySelector('table.data-table');
  const body = table && table.tBodies[0];
  if (!body || table.dataset.rowKeys) return;
  table.dataset.rowKeys = 'on';

  // A row must already be focusable before anything can focus it, and the
  // sorter re-appends these same elements rather than rebuilding them, so
  // this survives every reorder.
  for (const row of body.rows) row.tabIndex = -1;

  // What the reader can actually see: `filtertable.js` hides a row rather
  // than removing it, and stepping onto a hidden row would move focus to
  // nothing.
  const shown = () => [...body.rows].filter((row) => !row.hidden);

  const goTo = (row) => {
    if (!row) return;
    for (const other of body.rows) other.classList.remove('row-here');
    row.classList.add('row-here');
    row.focus();
  };

  region.addEventListener('keydown', (event) => {
    const steps = { ArrowDown: 1, ArrowUp: -1, End: Infinity,
                    // Home is a step of "all the way back" whichever row you
                    // are on, and the first row when you are on none — which
                    // is what a negative step of unbounded size already means
                    // once the index is clamped, except from nowhere at all.
                    Home: -Infinity };
    const rows = shown();
    if (!rows.length) return;
    const here = event.target.closest ? event.target.closest('tbody tr') : null;

    if (event.key === 'Enter' && here && event.target === here) {
      const control = here.querySelector('button, a');
      if (!control) return;
      event.preventDefault();
      control.click();
      return;
    }
    const step = steps[event.key];
    if (step === undefined) return;
    event.preventDefault();
    // Where the keyboard already is, or — arriving from the region itself —
    // the end the key was reaching for.
    const at = here ? rows.indexOf(here) : -1;
    const to = at >= 0 ? at + step
      : step === 1 ? 0                  // ArrowDown from the region: the top
      : step === -1 ? rows.length - 1   // ArrowUp from it: the bottom
      : step;                           // Home and End clamp to their own end
    goTo(rows[Math.min(rows.length - 1, Math.max(0, to))]);
  });
}

// Place a reveal against its cell, in viewport coordinates.
//
// Both surfaces now have one: the Inspector's chunk reveal and the board's
// settings reveal. Both hang off a cell inside a bounded, scrollable region —
// the Inspector's inside `.table-scroll`, the board's off a `position: sticky`
// cell inside the same — so an absolutely-positioned box is clipped at every
// width. `position: fixed` escapes the region, and then the box has to be put
// somewhere by hand, which is this.
//
// Flips above the cell when there is no room below, and never leaves the
// viewport sideways.
function placeReveal(cell, selector) {
  const reveal = cell && cell.querySelector(selector);
  // `position: static` is the narrow-width variant, which opens in place and
  // wants no insets at all.
  if (!reveal || getComputedStyle(reveal).position === 'static') return;
  const box = cell.getBoundingClientRect();
  const gap = 4;
  const below = box.bottom + gap;
  const height = reveal.offsetHeight;
  reveal.style.top = `${below + height <= window.innerHeight
    ? below : Math.max(gap, box.top - gap - height)}px`;
  reveal.style.left = `${Math.max(gap,
    Math.min(box.left, window.innerWidth - reveal.offsetWidth - gap))}px`;
}


// --- A second scrollbar, above the table -----------------------------------
// A wide table is scrolled with the bar under it, which on a long table is off
// the bottom of the screen: to move the columns you scroll the page down, drag,
// then scroll back up to read what moved. So the region gets a second bar above
// it, which is the same control — one holds a spacer as wide as the table has to
// scroll, and the two carry each other's `scrollLeft`.
//
// Hidden from assistive tech, deliberately: it is a duplicate handle on a region
// that is already reachable, announced and keyboard-scrollable in its own right,
// and a second entry in the tab order buys nothing but a stop that says nothing.
function mountScrollRail(region) {
  if (!region || region.previousElementSibling
      && region.previousElementSibling.classList.contains('scroll-rail')) return;
  const rail = document.createElement('div');
  rail.className = 'scroll-rail';
  rail.setAttribute('aria-hidden', 'true');
  const spacer = document.createElement('i');
  rail.appendChild(spacer);
  region.parentNode.insertBefore(rail, region);

  // The width the region can scroll, not the width of anything on screen: this
  // is what makes the two bars the same length and the same drag.
  const measure = () => {
    spacer.style.width = `${region.scrollWidth}px`;
    // A table that fits needs no bar at all, on either side of it.
    rail.hidden = region.scrollWidth <= region.clientWidth;
  };

  // One flag for both directions. Setting `scrollLeft` fires `scroll`, so
  // without it each bar would answer the other's answer.
  let syncing = false;
  const follow = (from, to) => () => {
    if (syncing) return;
    syncing = true;
    to.scrollLeft = from.scrollLeft;
    // Cleared after the event this assignment queues has been dispatched.
    requestAnimationFrame(() => { syncing = false; });
  };
  rail.addEventListener('scroll', follow(rail, region));
  region.addEventListener('scroll', follow(region, rail));

  // The table's width changes without the window's: a filter hides rows, the
  // pipeline column's content decides its own width, fonts arrive late.
  if (typeof ResizeObserver === 'function') {
    const watch = new ResizeObserver(measure);
    watch.observe(region);
    const table = region.querySelector('table');
    if (table) watch.observe(table);
  }
  window.addEventListener('resize', measure);
  measure();
  return measure;
}


// --- one explainer, read in two lengths ------------------------------------
// Every knob and every metric on these pages carries a sentence, and until now
// the only way to see it was to click a `!` and read the whole thing. Forty of
// those marks on one knob surface is a page speckled with punctuation, and the
// full paragraph is more than a reader wants for "what is Candidates?".
//
// So the explainer is read in two lengths, which is the pattern IBM's Carbon
// calls a *definition tooltip* (a term with a dotted underline; hover or focus
// shows a short definition, read-only) sitting in front of its *interactive*
// one (opened by a click, and it stays until dismissed). The brief comes from
// the service — the opening sentence of the same text, taken once server-side,
// so the two lengths cannot come to disagree.
//
// This half is the hover. The click is each page's own, because *where* the
// full text lands is a page decision the pages already make differently: the
// lab puts it after the whole field, the Inspector at the end of the metrics
// row so opening one cannot push the scores out of line.
//
// Rules taken from the pattern rather than invented here: a short delay on
// hover so the box does not flash at a mouse crossing the page, no delay at
// all on keyboard focus, `aria-describedby` while it is open so a screen
// reader gets the sentence, and — because a hover is not available to every
// reader — nothing is *only* reachable this way: the same trigger opens the
// full text with Enter.
const HELP_HOVER_MS = 140;
// Long enough to cross the gap between a trigger and the box under it, short
// enough that the box does not linger over the control it explains.
const HELP_LEAVE_MS = 160;
const HELP_TRIGGERS = '.why, .why-term';

const LabHelp = {
  // Each page sets these two: a topic key -> its one sentence, and the same
  // key -> the whole note. The defaults keep a page that never sets them
  // silent rather than broken.
  brief: () => '',
  full: () => '',
  box: null,
  trigger: null,
  timer: null,
  inside: false,
};

// A trigger says its own sentence if it carries one (a board row's error text
// has no topic to look up); otherwise the page resolves its topic; otherwise
// the first sentence of the full text is the honest fallback.
function helpBrief(trigger) {
  const own = trigger.dataset.brief;
  if (own) return own;
  const resolved = trigger.dataset.topic ? LabHelp.brief(trigger.dataset.topic) : '';
  if (resolved) return resolved;
  const full = trigger.dataset.help || '';
  return full.split(/(?<=[.!?])\s/)[0] || '';
}

function helpBox() {
  if (LabHelp.box) return LabHelp.box;
  const box = document.createElement('div');
  box.className = 'help-brief';
  box.id = 'help-brief';
  box.setAttribute('role', 'tooltip');
  box.hidden = true;
  // The box is a click target as well as a hover one: the reader who is
  // already looking at the sentence should be able to open the rest of it
  // where their eyes are, rather than travelling back to the trigger.
  box.addEventListener('mouseenter', () => { LabHelp.inside = true; });
  box.addEventListener('mouseleave', () => { LabHelp.inside = false; hideHelpBrief(); });
  box.addEventListener('click', () => {
    const trigger = LabHelp.trigger;
    hideHelpBrief(true);
    if (trigger) trigger.click();
  });
  document.body.appendChild(box);
  LabHelp.box = box;
  return box;
}

// Above the trigger when there is room, below it when there is not, and never
// off the side. Above by preference, because a knob's name sits directly on top
// of the knob: a box opening downwards lands on the control the reader is about
// to change, and they then have to wait for it to close before they can reach
// it. Same reasoning as `placeReveal`, and deliberately not the same function:
// that one places a box against a table cell inside a scroll region.
function placeHelpBrief(trigger, box) {
  const at = trigger.getBoundingClientRect();
  const gap = 6;
  const height = box.offsetHeight;
  const above = at.top - gap - height;
  box.style.top = `${above >= gap ? above : at.bottom + gap}px`;
  box.style.left = `${Math.max(gap,
    Math.min(at.left, window.innerWidth - box.offsetWidth - gap))}px`;
}

// Whether the click has anything left to say. A trigger carrying its own text
// answers for itself; otherwise the page resolves the topic.
function helpHasMore(trigger, brief) {
  const full = trigger.dataset.help
    || (trigger.dataset.topic ? LabHelp.full(trigger.dataset.topic) : '');
  return Boolean(full) && full.trim() !== brief.trim();
}

function showHelpBrief(trigger) {
  const text = helpBrief(trigger);
  if (!text) return;
  const box = helpBox();
  box.textContent = text;
  box.dataset.more = String(helpHasMore(trigger, text));
  box.hidden = false;
  LabHelp.trigger = trigger;
  trigger.setAttribute('aria-describedby', 'help-brief');
  placeHelpBrief(trigger, box);
}

function hideHelpBrief(now = false) {
  clearTimeout(LabHelp.timer);
  const close = () => {
    if (LabHelp.inside && !now) return;
    if (LabHelp.box) LabHelp.box.hidden = true;
    if (LabHelp.trigger) LabHelp.trigger.removeAttribute('aria-describedby');
    LabHelp.trigger = null;
  };
  if (now) close();
  else LabHelp.timer = setTimeout(close, HELP_LEAVE_MS);
}

function mountHelpBriefs() {
  document.addEventListener('mouseover', (event) => {
    const trigger = event.target.closest && event.target.closest(HELP_TRIGGERS);
    if (!trigger || trigger === LabHelp.trigger) return;
    clearTimeout(LabHelp.timer);
    LabHelp.timer = setTimeout(() => showHelpBrief(trigger), HELP_HOVER_MS);
  });
  document.addEventListener('mouseout', (event) => {
    if (event.target.closest && event.target.closest(HELP_TRIGGERS)) hideHelpBrief();
  });
  // No delay for a keyboard: a reader who tabbed here asked for it.
  document.addEventListener('focusin', (event) => {
    const trigger = event.target.closest && event.target.closest(HELP_TRIGGERS);
    if (trigger) showHelpBrief(trigger);
  });
  document.addEventListener('focusout', (event) => {
    if (event.target.closest && event.target.closest(HELP_TRIGGERS)) hideHelpBrief(true);
  });
  // The brief has said its piece the moment the full text opens.
  document.addEventListener('click', (event) => {
    if (event.target.closest && event.target.closest(HELP_TRIGGERS)) hideHelpBrief(true);
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') hideHelpBrief(true);
  });
  window.addEventListener('scroll', () => hideHelpBrief(true), { passive: true });
}

document.addEventListener('DOMContentLoaded', mountHelpBriefs);
