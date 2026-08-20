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
