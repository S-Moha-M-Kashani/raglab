"""The split plan: one ordered list of stages saying where a document is cut.

Two forms, two jobs. The *stored* form is a list of stage objects — what
`IndexConfig.fingerprint()` hashes and what an archive carries, explicit
enough that no parser sits in the trust path. The *typed* form is one line a
person writes in a sweep candidate or reads on a knob page:

    document / drift
    document / part
    document / role=user
    document / "\\n\\n" / "\\n" / ". "
    document / "\\n\\n" and role=assistant / part over-budget

Stages are separated by `/`, and each one subdivides what the stage before it
produced. `document` is always first. A stage is either a keyword (`part`,
`drift`) or atoms joined by one combinator — `or` (cut wherever any matches)
or `and` (cut only where every one holds), never both in one stage. An atom is
a quoted literal (a separator, with `\\n`, `\\t`, `\\"` and `\\\\` as JSON
writes them) or `label=value` (a boundary at every part carrying that label).
A `drift` stage may list topic-change markers as `drift or "…" or "…"`. A
trailing `always` or `over-budget` says when the stage applies; it is written
only when it differs from the kind's own default.
"""
import json

from .option_vocabularies import COMBINATORS, STAGE_KINDS, STAGE_WHEN

DOCUMENT = {'kind': 'document'}
# Structural stages cut regardless of size; a separator descends only into a
# piece that is still too big, which is what makes a coarse-to-fine list of
# separators leave a short paragraph whole.
DEFAULT_WHEN = {'part': 'always', 'label': 'always', 'drift': 'always',
                'separator': 'over-budget'}
# The lab's own default: cut where the subject changes.
DEFAULT = (DOCUMENT, {'kind': 'drift', 'markers': (), 'when': 'always'})



# --- the stored form ------------------------------------------------------

def normalize(stages) -> tuple[dict, ...]:
    """One payload per plan: every stage carries its whole shape, lists are
    tuples, and a `when` left out is the kind's own default. Lenient about
    what it cannot fix — `problems()` is where a bad plan is named."""
    out = []
    for stage in stages or ():
        if not isinstance(stage, dict):
            out.append(stage)
            continue
        kind = stage.get('kind')
        fixed = {'kind': kind}
        if kind in DEFAULT_WHEN:
            fixed['when'] = stage.get('when') or DEFAULT_WHEN[kind]
        if kind in ('label', 'separator'):
            fixed['atoms'] = tuple(dict(a) if isinstance(a, dict) else a
                                   for a in stage.get('atoms') or ())
            fixed['join'] = stage.get('join') or 'or'
        if kind == 'drift':
            fixed['markers'] = tuple(stage.get('markers') or ())
        out.append(fixed)
    return tuple(out)


def is_label(atom) -> bool:
    return isinstance(atom, dict) and 'label' in atom


def is_text(atom) -> bool:
    return isinstance(atom, dict) and 'text' in atom


def cuts_text(stage: dict) -> bool:
    """A stage with a literal in it cuts inside a part, after which no part
    identity remains for a later stage to read."""
    return stage.get('kind') == 'separator'


def needs_parts(stage: dict) -> bool:
    """Drift compares parts; a label boundary selects them; a part stage is them."""
    return stage.get('kind') in ('part', 'label', 'drift')


def problems(stages, label_fields: dict | None = None) -> list[str]:
    """Everything wrong with a plan, each naming what was wrong. `label_fields`
    is the selected corpus's own declaration table, so a label boundary on a
    label that corpus never declares is refused here, before a build."""
    bad: list[str] = []
    stages = tuple(stages or ())
    if not stages or stages[0] != DOCUMENT:
        bad.append('split_plan must begin with the document stage — no chunk '
                   'ever spans two documents, and the plan says so')
    cut = False
    for number, stage in enumerate(stages):
        if not isinstance(stage, dict) or stage.get('kind') not in STAGE_KINDS:
            bad.append(f'split_plan stage {number}: unknown stage {stage!r} '
                       f'(expected one of {", ".join(STAGE_KINDS)})')
            continue
        kind = stage['kind']
        if kind == 'document' and number > 0:
            bad.append(f'split_plan stage {number}: the document stage is the '
                       'first stage and no other')
            continue
        if kind == 'document':
            continue
        if stage.get('when') not in STAGE_WHEN:
            bad.append(f'split_plan stage {number}: when must be one of '
                       f'{", ".join(STAGE_WHEN)}, not {stage.get("when")!r}')
        if kind in ('label', 'separator'):
            bad.extend(_atom_problems(number, stage, label_fields))
        if kind == 'drift' and not all(isinstance(m, str) and m
                                       for m in stage.get('markers', ())):
            bad.append(f'split_plan stage {number}: drift markers must be '
                       'non-empty strings')
        if needs_parts(stage) and cut:
            what = ('drift compares parts through their embeddings'
                    if kind == 'drift' else
                    'a label boundary cuts at part boundaries')
            bad.append(f'split_plan stage {number}: a {kind} stage cannot '
                       f'follow a separator stage — {what}, and no parts '
                       'remain once text has been cut')
        if kind in ('label', 'part', 'drift'):
            bad.extend(_declared(number, stage, label_fields))
        cut = cut or cuts_text(stage)
    return bad


def _atom_problems(number: int, stage: dict, label_fields) -> list[str]:
    bad = []
    if stage.get('join') not in COMBINATORS:
        bad.append(f'split_plan stage {number}: a stage combines its atoms '
                   f'with "or" or with "and", not {stage.get("join")!r}')
    atoms = stage.get('atoms') or ()
    if not atoms:
        bad.append(f'split_plan stage {number}: a {stage["kind"]} stage needs '
                   'at least one atom')
    for atom in atoms:
        if is_text(atom):
            if not isinstance(atom['text'], str) or not atom['text']:
                bad.append(f'split_plan stage {number}: a separator must be a '
                           'non-empty string')
            elif stage['kind'] == 'label':
                bad.append(f'split_plan stage {number}: a label stage cuts at '
                           f'part boundaries; the literal {atom["text"]!r} '
                           'belongs in a separator stage')
        elif is_label(atom):
            if (not isinstance(atom['label'], str) or not atom['label']
                    or not isinstance(atom.get('value'), str)):
                bad.append(f'split_plan stage {number}: a label boundary is '
                           f'{{"label": name, "value": value}}, not {atom!r}')
        else:
            bad.append(f'split_plan stage {number}: unknown atom {atom!r} — '
                       'a literal is {"text": …}, a label boundary '
                       '{"label": …, "value": …}')
    if stage['kind'] == 'separator' and not any(is_text(a) for a in atoms):
        bad.append(f'split_plan stage {number}: a separator stage needs a '
                   'literal to cut on')
    return bad


def _declared(number: int, stage: dict, label_fields) -> list[str]:
    """A label boundary on a label the corpus never declares at the part level
    would silently cut nothing; it is refused by name instead."""
    if label_fields is None or stage.get('kind') != 'label':
        return []
    bad = []
    for atom in stage.get('atoms') or ():
        if not is_label(atom) or not isinstance(atom.get('label'), str):
            continue
        definition = label_fields.get(atom['label'])
        if not definition or 'part' not in (definition.get('applies_to') or []):
            bad.append(f'split_plan stage {number}: the selected corpus '
                       f'declares no part-level label {atom["label"]!r}')
            continue
        values = definition.get('values')
        if values and atom.get('value') not in values:
            bad.append(f'split_plan stage {number}: {atom["label"]!r} takes '
                       f'one of {", ".join(map(str, values))}, not '
                       f'{atom.get("value")!r}')
    return bad


# --- the typed form ---------------------------------------------------------

def _tokens(line: str) -> list[str]:
    """Words, quoted literals (kept with their quotes) and `/`, whitespace
    dropped — which is what makes two spellings differing only in spacing one
    plan."""
    out: list[str] = []
    i = 0
    while i < len(line):
        ch = line[i]
        if ch.isspace():
            i += 1
        elif ch == '/':
            out.append('/')
            i += 1
        elif ch == '"':
            j = i + 1
            while j < len(line) and line[j] != '"':
                j += 2 if line[j] == '\\' else 1
            if j >= len(line):
                raise ValueError(f'split_plan: unclosed quote in {line!r}')
            out.append(line[i:j + 1])
            i = j + 1
        else:
            j = i
            while j < len(line) and not line[j].isspace() and line[j] not in '/"':
                j += 1
            out.append(line[i:j])
            i = j
    return out


def _atom(word: str) -> dict:
    if word.startswith('"'):
        return {'text': json.loads(word)}
    name, sep, value = word.partition('=')
    if not sep or not name:
        raise ValueError(f'split_plan: {word!r} is neither a quoted separator '
                         'nor a label=value boundary')
    return {'label': name, 'value': value}


def _stage(words: list[str]) -> dict:
    if not words:
        raise ValueError('split_plan: an empty stage')
    when = None
    if len(words) > 1 and words[-1] in STAGE_WHEN:
        when, words = words[-1], words[:-1]
    if words[0] == 'document':
        if len(words) > 1 or when:
            raise ValueError('split_plan: the document stage takes nothing')
        return dict(DOCUMENT)
    if words[0] == 'part':
        if len(words) > 1:
            raise ValueError('split_plan: the part stage takes no atoms')
        return {'kind': 'part', 'when': when or DEFAULT_WHEN['part']}
    if words[0] == 'drift':
        markers = _joined(words[1:], only='or', after='drift')
        return {'kind': 'drift', 'markers': tuple(a['text'] for a in markers),
                'when': when or DEFAULT_WHEN['drift']}
    atoms = [_atom(words[0])]
    join = 'or'
    rest = words[1:]
    if rest:
        join = rest[0]
        atoms.extend(_joined(rest, only=join, after=words[0]))
    kind = 'label' if all(is_label(a) for a in atoms) else 'separator'
    return {'kind': kind, 'atoms': tuple(atoms), 'join': join,
            'when': when or DEFAULT_WHEN[kind]}


def _joined(words: list[str], only: str, after: str) -> list[dict]:
    """`or x or y` or `and x and y`, one combinator throughout."""
    if only not in COMBINATORS:
        raise ValueError(f'split_plan: expected "or" or "and" after {after}, '
                         f'not {only!r}')
    atoms = []
    for position, word in enumerate(words):
        if position % 2 == 0:
            if word != only:
                raise ValueError(f'split_plan: a stage takes "or" or "and", '
                                 f'not both — found {word!r} after {only!r}')
        else:
            atoms.append(_atom(word))
    if len(words) % 2:
        raise ValueError(f'split_plan: an atom is missing after {words[-1]!r}')
    return atoms


def parse(line: str) -> tuple[dict, ...]:
    """The typed form to the stored form. Raises ValueError on a line that
    is not a plan at all; a plan that is well-formed but impossible is left
    for `problems()` to refuse by name."""
    stages, current = [], []
    for token in _tokens(line):
        if token == '/':
            stages.append(_stage(current))
            current = []
        else:
            current.append(token)
    stages.append(_stage(current))
    return normalize(stages)


def _quoted(literal: str) -> str:
    return json.dumps(literal, ensure_ascii=False)


def _word(atom: dict) -> str:
    if is_text(atom):
        return _quoted(atom['text'])
    return f'{atom["label"]}={atom["value"]}'


def text(stages) -> str:
    """The stored form as the one line a person reads or types."""
    out = []
    for stage in normalize(stages):
        kind = stage.get('kind')
        if kind == 'document':
            words = ['document']
        elif kind == 'part':
            words = ['part']
        elif kind == 'drift':
            words = ['drift']
            for marker in stage['markers']:
                words += ['or', _quoted(marker)]
        else:
            words = []
            for atom in stage.get('atoms', ()):
                if words:
                    words.append(stage['join'])
                words.append(_word(atom))
            words = words or [kind]
        if kind in DEFAULT_WHEN and stage.get('when') != DEFAULT_WHEN[kind]:
            words.append(stage['when'])
        out.append(' '.join(words))
    return ' / '.join(out)


def short(stages) -> str:
    """The plan without its always-present first stage — what a board column
    or a run label has room for. `document` when that is all there is."""
    stages = normalize(stages)
    return text(stages[1:]) if len(stages) > 1 else 'document'
