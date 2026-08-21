// Shared between the panel (:9002) and the Inspector (:9003) — loaded before
// each page's own script, the way sorttable.js already is. Holds only what
// the two pages' scripts, written months apart, turned out to need identically.

// The stricter of the two copies this page used to carry separately: escapes
// `"` as well as `&<>`. In a text node `&quot;` renders as `"`, so nothing
// looks different there; in an attribute value it closes a latent injection.
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
// :9002 and :9003 are separate origins, so neither surface can read the
// other's storage and the choice does not travel between them. That is a
// browser rule rather than a decision — the Inspector remembers its own.
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

document.addEventListener('DOMContentLoaded', () => {
  mountThemeControl(document.getElementById('theme-control'));
});

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
