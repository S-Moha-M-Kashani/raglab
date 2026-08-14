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
