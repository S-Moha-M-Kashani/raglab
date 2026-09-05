"""The frontend files this service makes public.

One route serves all of them (`install_assets`), reading the allowlist below.
A file sitting in `frontend/` that this table does not name is a 404, which is
what keeps `panel.html` reachable at `/` and nowhere else, and what stops a
request path walking out of the folder."""
from raglab.dashboard.service_route_plumbing import Asset, install_assets

# Every file this service serves to a browser, and why it is served here. One
# route reads this table (`install_assets`); a file in `frontend/` that is not
# named here is a 404, so `panel.html` and `leaderboard.html` keep the single
# address each is reached at.
ASSETS = {
    '/': Asset('panel.html', None,
               'The Laboratory itself, at the root — the only address it has, '
               'so its filename is not a second one.'),
    '/panel.css': Asset(
        'panel.css', 'text/css',
        "The panel's style, extracted from panel.html's <style> block."),
    '/panel.js': Asset(
        'panel.js', 'application/javascript',
        "The panel's script, extracted from panel.html's <script> block."),
    '/leaderboard': Asset(
        'leaderboard.html', None,
        'The cross-run surface: what earlier runs said, kept off the lab page '
        'where the knobs live.'),
    '/leaderboard.js': Asset(
        'leaderboard.js', 'application/javascript',
        "The leaderboard surface's script — it renders what /api/leaderboard "
        'serves and re-derives no rank of its own.'),
    '/dataset': Asset(
        'dataset.html', None,
        'The corpus viewer: the documents, parts and questions a run is '
        'measured against, read-only and off the lab page.'),
    '/dataset.js': Asset(
        'dataset.js', 'application/javascript',
        "The corpus viewer's script — it renders what "
        '/api/dataset-content serves and derives no reading of its own.'),
    '/dataset.css': Asset(
        'dataset.css', 'text/css',
        "The corpus viewer's own rules: the readings row, the parts panel "
        'and the raw tree, which no other surface has.'),
    '/tokens.css': Asset(
        'tokens.css', 'text/css',
        'The design tokens shared with the Inspector, so a colour cannot '
        'drift apart on either page.'),
    '/chrome.css': Asset(
        'chrome.css', 'text/css',
        'The bar and surface switcher shared with the Inspector, so the top '
        'of a page means one thing on both ports.'),
    '/lab.js': Asset(
        'lab.js', 'application/javascript',
        'The utilities shared with the Inspector, so a name like escapeHtml '
        'has one behaviour, not two.'),
    '/sorttable.js': Asset(
        'sorttable.js', 'application/javascript',
        'The column sorter, shared with the Inspector — one of three static '
        'files served outside the one page.'),
    '/filtertable.js': Asset(
        'filtertable.js', 'application/javascript',
        "The leaderboard's row filter, which reads a cell with the sorter's "
        'own parser rather than a second one.'),
    '/widget.css': Asset(
        'widget.css', 'text/css',
        "The widget's own rules, served to all three surfaces — the helper is "
        "not the Laboratory's, so its sheet is not panel.css."),
    '/widget.js': Asset(
        'widget.js', 'application/javascript',
        'The widget itself. One file, three pages: it builds its own markup, '
        'so a surface gains the helper by loading this and nothing else.'),
    '/archive_io.js': Asset(
        'archive_io.js', 'application/javascript',
        'The versioned archive codec, loaded before the Panel integration.'),
    '/experiment_handoff.js': Asset(
        'experiment_handoff.js', 'application/javascript',
        'The board-to-Laboratory handoff, loaded by both pages: the board '
        'writes the slot, the panel decides which recorded knobs this '
        'installation can serve.'),
}


def register(app, context) -> None:
    """The allowlist is the whole registration: no state, so nothing off the
    context is read here."""
    install_assets(app, ASSETS)
