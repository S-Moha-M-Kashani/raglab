// The browser codec over the ladder: knobs, then what was indexed, then what
// was retrieved, then what was generated — and beside that last one, the same
// judged experiment with no trace kept, which is the shape every evaluation
// recorded before the export route existed is in.
//
// This is the export a reader actually clicks — `ArchiveIO.completed` is what
// `exportArchive` calls, and `ArchiveIO.parse` is what an imported file goes
// through before a single control moves. The fixtures are not written here:
// they are generated from the lab's own dataclasses by `archive_examples.py`
// and handed over as JSON, so a knob added to the config and forgotten in the
// browser template fails on this side too, without a second list to maintain.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runInNewContext } from 'node:vm';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const SOURCE = readFileSync(join(HERE, '../frontend/archive_io.js'), 'utf8');
const ArchiveIO = runInNewContext(SOURCE + '\n;ArchiveIO', {});
const plain = (value) => JSON.parse(JSON.stringify(value));

// Written by `test_archive_io.py`, which is the only way this file is run.
const LADDER = JSON.parse(readFileSync(process.env.RAGLAB_LADDER, 'utf8'));

// The same reading of an archive the Python side makes, in the same terms.
const contents = (value) => {
  if (!value.evaluation) return { ...LADDER.carried.settings };
  const { inspector, result } = value.evaluation;
  const traces = inspector.traces;
  return {
    evaluation: true,
    sessions: inspector.dataset.corpus.sessions.length,
    questions: inspector.dataset.ground_truth.questions.length,
    chunks: inspector.chunks_by_session.reduce((n, g) => n + g.chunks.length, 0),
    summaries: inspector.summaries.length,
    traces: traces.length,
    candidates: traces.reduce((n, t) => n + t.trace.candidates.length, 0),
    rows: result.rows.length,
    answers: result.rows.filter((row) => row.answer).length,
    judged: Boolean((result.ragas || {}).metrics),
  };
};

// Every knob as `group.field`, so a diff names the knob rather than the config.
const knobs = (config) => {
  const flat = { label: config.label };
  for (const group of ['index', 'retrieval', 'generation']) {
    for (const [field, value] of Object.entries(config[group])) {
      flat[`${group}.${field}`] = value;
    }
  }
  return flat;
};

// The premise the rest rests on, asserted rather than assumed: no knob in the
// fixture is at its default, so a knob the codec drops cannot read back correct.
test('the ladder fixture moves every knob off its default', () => {
  const shifted = knobs(LADDER.config);
  const defaults = knobs(LADDER.defaults);
  assert.deepEqual(Object.keys(shifted).sort(), Object.keys(defaults).sort());
  const unmoved = Object.keys(shifted).filter(
    (knob) => JSON.stringify(shifted[knob]) === JSON.stringify(defaults[knob]));
  assert.deepEqual(unmoved, [],
    `these knobs are still at their default, so a round trip that dropped them `
    + `would look correct: ${unmoved.join(', ')}`);
});

for (const rung of LADDER.rungs) {
  test(`exporting the ${rung.name} rung writes every knob and every stage`, () => {
    const exported = rung.result === null
      ? ArchiveIO.settings(rung.config, rung.ui)
      : ArchiveIO.completed(rung.config, rung.ui, rung.result, rung.evidence);

    // An unrun experiment exports its knobs and must not grow readings.
    assert.equal('evaluation' in exported, rung.result !== null);
    assert.deepEqual(knobs(plain(exported.settings.config)), knobs(LADDER.config));
    assert.deepEqual(plain(exported.settings.ui), LADDER.ui);
    assert.deepEqual(contents(plain(exported)), LADDER.carried[rung.name]);

    // The whole file: the browser must write the archive the lab's own codec
    // would, or the two halves of this feature disagree about the format.
    assert.deepEqual(plain(exported), rung.archive);
  });

  test(`importing the ${rung.name} rung restores every knob and every stage`, () => {
    // Through the text, not the object: this is the file path, and what
    // `stringify` drops on the way out `parse` can never put back.
    const imported = ArchiveIO.parse(ArchiveIO.stringify(rung.archive));
    assert.deepEqual(knobs(plain(imported.settings.config)), knobs(LADDER.config));
    assert.deepEqual(plain(imported.settings.ui), LADDER.ui);
    assert.deepEqual(contents(plain(imported)), LADDER.carried[rung.name]);
    assert.deepEqual(plain(imported), rung.archive);
    assert.ok(ArchiveIO.equal(imported, rung.archive));
  });
}

// The branch rung is the whole reason the codecs stopped requiring one trace
// per row, so the ladder also pins what the relaxation did *not* buy: a trace
// list that runs in a different order from the selection is still a quiet claim
// about the run, and is still refused, in the browser as in Python.
test('traces may be absent, but the ones present keep the run\'s order', () => {
  const retrieved = LADDER.rungs.find((rung) => rung.name === 'retrieved');
  const scored = LADDER.rungs.find((rung) => rung.name === 'scored-without-traces');
  assert.equal(scored.archive.evaluation.inspector.traces.length, 0);
  assert.equal(scored.archive.evaluation.result.rows.length, 2);
  assert.doesNotThrow(() => ArchiveIO.normalize(plain(scored.archive)));

  const partial = plain(retrieved.archive);
  partial.evaluation.inspector.traces = [partial.evaluation.inspector.traces[0]];
  assert.doesNotThrow(() => ArchiveIO.normalize(partial));

  const reordered = plain(retrieved.archive);
  reordered.evaluation.inspector.traces.reverse();
  assert.throws(() => ArchiveIO.normalize(reordered), /traces.*must follow/);
});
