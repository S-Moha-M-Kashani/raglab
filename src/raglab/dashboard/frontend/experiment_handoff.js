// The leaderboard → Laboratory handoff. No DOM, no network, like `archive_io`.
//
// The board's open button does two things with one experiment: it pins the
// Inspector to it, and it hands the same experiment to the Laboratory so the
// knobs there are that experiment's. The board cannot do the second itself —
// only the panel holds `/api/options`, so only the panel can tell a knob this
// installation serves from one it merely recorded. So the board writes an *id*
// into one slot and the panel takes it from there.
//
// `reconcile` is the rule that keeps the second half honest. An unserved value
// written into a `<select>` with no such option reads back as '', and the lab
// would then build under a config that is neither the experiment's nor the
// reader's — a row lying about what produced it, one step earlier than the row.
// So an unserved knob is left where it was and *named*. The naming is the
// contract: it is the whole difference between "these are experiment X's
// settings" and a quiet substitution.
//
// The archive import refuses such a config outright (`ArchiveIO.transact`), and
// that is right for a file someone chose to import: it either arrives intact or
// it does not arrive. Opening a row of this lab's own board is the other case —
// the reader is already looking at the experiment and wants the panel nearer to
// it — so this path applies what it can and says what it could not.
const ExperimentHandoff = (() => {
  const KEY = 'lodestar:raglab-open-experiment';
  const GROUPS = Object.freeze(['index', 'retrieval', 'generation']);

  // --- the slot -------------------------------------------------------------

  // `at` comes from the caller rather than a clock in here, to keep this module
  // free of anything but its arguments — and it is not enough on its own. A
  // `storage` event fires on a *change*, so two clicks on one row that wrote the
  // same bytes would leave an already-open Laboratory tab hearing only the
  // first; a double-click is exactly that, inside one millisecond. So the write
  // is compared against what is standing in the slot and nudged past it, which
  // makes "different from last time" a property of the slot rather than a bet
  // on the clock's resolution.
  const offer = (storage, experimentId, at) => {
    const payload = (stamp) =>
      JSON.stringify({ experiment_id: experimentId, at: stamp });
    try {
      const stamp = Number(at) || 0;
      const written = payload(stamp);
      storage.setItem(KEY,
        storage.getItem(KEY) === written ? payload(stamp + 1) : written);
    } catch (e) { /* private browsing, a full quota: not worth a dead button */ }
  };

  // Taken once. The knobs it sets are remembered by the panel like any other
  // knobs, so a slot left behind would re-announce the same experiment on every
  // reload — a notice about something the reader did days ago.
  const taken = (storage) => {
    let held = null;
    try {
      held = JSON.parse(storage.getItem(KEY) || 'null');
      storage.removeItem(KEY);
    } catch (e) {
      // Hand-edited, or written by an older shape of this page. Forgetting it
      // is right; refusing to boot over it is not.
      try { storage.removeItem(KEY); } catch (ignored) { /* nothing to do */ }
      return null;
    }
    if (!held || typeof held !== 'object' || !held.experiment_id) return null;
    return { experiment_id: held.experiment_id, at: held.at };
  };

  // --- what this lab can serve ----------------------------------------------

  // One rule per knob the panel constrains, asked in the order that produces
  // the sentence a reader can act on: the corpus first, because a config
  // applied against the wrong corpus is not that experiment at all.
  function unservedReason(path, value, served) {
    if (path === 'index.dataset') {
      return (served.datasets || []).includes(value) ? '' : 'not installed here';
    }
    const choices = (served.choices || {})[path];
    if (choices) {
      return choices.includes(value) ? '' : 'not served by this lab';
    }
    const catalogue = (served.models || {})[path];
    if (catalogue) {
      // No model is a state a stage is allowed to be in — the panel's own empty
      // option — and not a model this lab fails to serve.
      if (!value) return '';
      // Each catalogue carries the reason it would refuse, because the two
      // refuse for different reasons: the chat models are what *this backend
      // mode* offers, the embedding models are what is installed here at all.
      // One reason for both sends half the readers to change the wrong thing.
      return (catalogue.ids || []).includes(value) ? '' : catalogue.reason;
    }
    const range = (served.ranges || {})[path];
    if (range) {
      const low = range.min !== null && range.min !== undefined
        && Number(value) < range.min;
      const high = range.max !== null && range.max !== undefined
        && Number(value) > range.max;
      if (!low && !high) return '';
      return `outside this panel’s range, ${rangeSaid(range)}`;
    }
    // A knob no control shows — `rrf_k`, `agentic_weights`, `max_context_chars`.
    // The panel carries these through untouched (`UNSHOWN`), so dropping them
    // here would quietly re-run the experiment under this lab's defaults.
    return '';
  }

  const rangeSaid = (range) => {
    const hasLow = range.min !== null && range.min !== undefined;
    const hasHigh = range.max !== null && range.max !== undefined;
    if (hasLow && hasHigh) return `${range.min} to ${range.max}`;
    return hasLow ? `at least ${range.min}` : `at most ${range.max}`;
  };

  // --- reconcile ------------------------------------------------------------

  // `recorded` is the experiment's config as the ledger and the run file jointly
  // recorded it, which for a ledger-only row names as few as six knobs; `current`
  // is what the panel has on screen, which is what every knob the record does not
  // name — and every knob it names that this lab cannot serve — stays at.
  function reconcile(recorded, current, served) {
    const config = { label: current.label || '' };
    for (const group of GROUPS) {
      config[group] = Object.assign({}, current[group] || {});
    }
    if (typeof (recorded || {}).label === 'string') config.label = recorded.label;

    const unserved = [];
    const set = [];
    for (const group of GROUPS) {
      const knobs = (recorded || {})[group] || {};
      for (const [knob, value] of Object.entries(knobs)) {
        const path = `${group}.${knob}`;
        const reason = unservedReason(path, value, served);
        if (reason) unserved.push({ path, value, reason });
        else { config[group][knob] = value; set.push(path); }
      }
    }
    // The corpus first, whatever order the record listed its knobs in. A config
    // applied against the wrong corpus is not that experiment at all, whatever
    // else survived, so it must not be a line the reader has to go and find —
    // and it is the refusal the archive path should give first for the same
    // reason, since it reads `unserved[0]`.
    unserved.sort((a, b) => (a.path === 'index.dataset' ? -1 : 0)
      - (b.path === 'index.dataset' ? -1 : 0));
    return { config, unserved, set };
  }

  // --- what the reader is told ----------------------------------------------

  // The statement that keeps the whole handoff honest, so it lives with the
  // reconciliation it describes rather than in the page that displays it. Three
  // things can be true at once and each is a separate sentence: how much of the
  // panel is now this experiment, which knobs this installation could not serve,
  // and whether the record itself was ever complete.
  function notice(record, out) {
    const when = String(record.started_at || '').slice(0, 16);
    const named = [record.kind || 'experiment', when].filter(Boolean).join(' · ');
    const head = `Laboratory settings are now experiment ${record.experiment_id}`
      + `${named ? ` (${named})` : ''}`
      + `${record.dataset ? ` on ${record.dataset}` : ''}.`;

    // A count of nothing reads as an accident rather than as a fact about the
    // record, so the empty case says what it is instead of counting to zero.
    const moved = out.set.length
      ? `${out.set.length} knob${out.set.length === 1 ? '' : 's'} set.`
      : 'It recorded no settings, so no knob changed.';

    const said = [head, moved];
    if (out.unserved.length) {
      const count = out.unserved.length;
      said.push(`${count} could not be set here and ${count === 1
        ? 'is' : 'are'} unchanged: ` + out.unserved.map((row) =>
        `${row.path} = ${String(row.value)} (${row.reason})`).join('; ') + '.');
    }
    // Why the panel is only partly this experiment can be the record rather than
    // the installation, and the two must never be told in one sentence: an index
    // build or a retrieval has no run file to have recorded more, and neither
    // does an evaluation whose run file has since been deleted.
    if (record.source === 'ledger' && out.set.length) {
      said.push('This row has no run file, so the ledger recorded only the '
        + `${out.set.length} knob${out.set.length === 1 ? '' : 's'} it names; `
        + 'every other knob is unchanged.');
    }
    return said.join(' ');
  }

  return Object.freeze({ KEY, GROUPS, offer, taken, reconcile, notice });
})();
