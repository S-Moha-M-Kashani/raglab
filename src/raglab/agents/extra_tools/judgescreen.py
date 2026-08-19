"""Screen a model before letting it judge, on a held-out task with known answers built by `_mutate_number`: the context is dated and the claim is one sentence.
Degeneracy and schema adherence are scored separately (`score`'s `degenerate`, and `parsed` since RAGAS retries malformed JSON) because the two are fixed differently.
The judge must be chosen by its score here **before** looking at the leaderboard it would produce — that is judge-shopping.
"""
import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from raglab.rag_components.retrieval import farsi_text_normalizer as textnorm

from raglab.corpora import diary_corpus_loader as corpus
from raglab.configuration.lab_config import ROOT, load_lab_settings
from raglab.llm_backends.chat_model_factory import judge_llm

# Anchored to ROOT, not counted in `parents[n]` hops from this file: the hop
# count broke the day the module moved into agents/extra_tools/, pointing it at src/.
SCREENS_DIR = ROOT / '.screens'

# Shaped like RAGAS's NLIStatementOutput, so this screens the format RAGAS
# actually asks for rather than a flatter one.
PROMPT = (
    'You decide whether each statement is supported by the context.\n'
    'The context is in Persian. Use ONLY the context — not general knowledge.\n'
    'Reply with JSON only, no prose, in exactly this shape:\n'
    '{"statements": [{"statement": "<the statement>", "verdict": 0 or 1, '
    '"reason": "<one short sentence>"}]}\n'
    'verdict 1 = the context supports the statement. '
    'verdict 0 = it does not, or it contradicts it.')


@dataclass
class Item:
    """One claim, its context, and the verdict the ground truth already implies."""
    id: str
    question_id: str
    context: list[str]
    claim: str
    supported: bool
    overlap: float          # claim/context token overlap, for the inversion check


@dataclass
class Call:
    """What went to the model and what came back, kept verbatim rather than
    summarised — a bare accuracy cannot be re-read to see how a model failed."""
    item_id: str
    supported: bool
    verdict: int | None     # None = the reply could not be parsed
    parsed: bool
    seconds: float
    prompt: str
    reply: str
    usage: dict = field(default_factory=dict)


def _overlap(claim: str, context: list[str]) -> float:
    """Fraction of the claim's content tokens that appear in the context."""
    claim_tokens = set(textnorm.tokens(claim))
    if not claim_tokens:
        return 0.0
    context_tokens = set(textnorm.tokens(' '.join(context)))
    return len(claim_tokens & context_tokens) / len(claim_tokens)


NUMERAL = re.compile(r'[0-9۰-۹٠-٩]+')
# Offsets tried in order until one lands on a number the context does not state.
# Small and odd: a mutated year has to stay a plausible year, or the judge is
# being asked to spot nonsense rather than a fabricated fact.
OFFSETS = (7, 3, 11, 23, 41)


def _mutate_number(claim: str, context: list[str]) -> str | None:
    """The claim with one numeral changed to one the context never states, or
    None rather than guessing — a "wrong" number the context does mention would
    mislabel a claim as unsupported and disqualify the judge that got it right."""
    normalised_context = textnorm.normalize(' '.join(context))
    for match in NUMERAL.finditer(claim):
        original = textnorm.normalize(match.group())
        if not original.isdigit() or len(original) > 4:
            continue
        # Only mutate a number the context actually states: changing one the
        # context never mentions produces a claim that was already unsupported,
        # which measures nothing about the change.
        if original not in normalised_context:
            continue
        for offset in OFFSETS:
            candidate = str(int(original) + offset)
            if candidate not in normalised_context:
                return claim[:match.start()] + candidate + claim[match.end():]
    return None


def dated_context(sessions: dict, question: dict) -> list[str]:
    """The cited evidence messages, each carrying the date of its session — as
    `IndexConfig.contextual` does for the real pipeline."""
    lines: list[str] = []
    for evidence in question.get('evidence', []):
        session = sessions.get(evidence['session_id'])
        if not session:
            continue
        for index in evidence.get('message_indices', []):
            if 0 <= index < len(session['messages']):
                lines.append(f"[{session['date']}] "
                             f"{session['messages'][index]['content']}")
    if lines:
        return lines
    return [f"[{ev.get('session_id', '')[:10]}] {ev['quote']}"
            for ev in question.get('evidence', [])]


def _anchored_sentence(answer: str, context: list[str]) -> str | None:
    """One sentence of the answer containing a number the context also states —
    the anchor the unsupported twin will change."""
    normalised = textnorm.normalize(' '.join(context))
    candidates = [s for s in textnorm.sentences(answer)
                  if any(textnorm.normalize(n) in normalised
                         for n in NUMERAL.findall(s)
                         if textnorm.normalize(n).isdigit())]
    if not candidates:
        return None
    # The best-supported candidate, so unrelated extra clauses lose to a cleaner one.
    return max(candidates, key=lambda s: _overlap(s, context))


def build_items(ground_truth: dict, sessions: dict, pairs: int = 6) -> list[Item]:
    """`pairs` supported claims and `pairs` unsupported ones, perfectly balanced —
    a degenerate judge then scores exactly 0.5 rather than a misleadingly
    respectable accuracy. A question yielding no anchored sentence or clean
    mutation is skipped whole, so the classes stay equal in size and wording."""
    usable = [q for q in ground_truth['questions']
              if q.get('answerable') and q.get('evidence') and q.get('answer_fa')]
    items: list[Item] = []
    for question in usable:
        if len(items) >= pairs * 2:
            break
        context = dated_context(sessions, question)
        if not context:
            continue
        claim = _anchored_sentence(question['answer_fa'], context)
        if not claim:
            continue
        mutated = _mutate_number(claim, context)
        if not mutated:
            continue
        items.append(Item(id=f"{question['id']}-yes", question_id=question['id'],
                          context=context, claim=claim, supported=True,
                          overlap=_overlap(claim, context)))
        items.append(Item(id=f"{question['id']}-no", question_id=question['id'],
                          context=context, claim=mutated, supported=False,
                          overlap=_overlap(mutated, context)))
    return items


def _verdict(reply: str) -> int | None:
    """The verdict out of a JSON reply, or None if the shape was not produced.
    Tolerates a fenced code block (a formatting habit, not a failure to answer),
    but not a missing `verdict` — RAGAS counts that as a parse failure too."""
    text = (reply or '').strip()
    if text.startswith('```'):
        text = text.split('```')[1] if '```' in text[3:] else text[3:]
        text = text.removeprefix('json').strip()
    start, end = text.find('{'), text.rfind('}')
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start:end + 1])
    except Exception:
        return None
    statements = payload.get('statements')
    if not isinstance(statements, list) or not statements:
        return None
    verdict = statements[0].get('verdict') if isinstance(statements[0], dict) else None
    if isinstance(verdict, bool):
        return int(verdict)
    if isinstance(verdict, (int, float)) and int(verdict) in (0, 1):
        return int(verdict)
    if isinstance(verdict, str) and verdict.strip() in ('0', '1'):
        return int(verdict.strip())
    return None


def ask(llm, item: Item) -> Call:
    user = ('Context:\n' + '\n'.join(f'- {line}' for line in item.context)
            + f'\n\nStatement:\n{item.claim}')
    started = time.time()
    try:
        turn = llm.invoke([{'role': 'system', 'content': PROMPT},
                           {'role': 'user', 'content': user}])
        reply = turn.content or ''
        usage = dict(turn.usage_metadata or {})
    except Exception as error:
        # A data point, not a crash: finding out a model can't do this is the job.
        reply, usage = f'ERROR: {error}', {}
    verdict = _verdict(reply)
    return Call(item_id=item.id, supported=item.supported, verdict=verdict,
                parsed=verdict is not None, seconds=round(time.time() - started, 2),
                prompt=user, reply=reply, usage=usage)


def score(calls: list[Call]) -> dict:
    """The screen's verdict on the judge.

    `degenerate` is the field that decides, not `accuracy`: a model that answers
    the same way every time is unusable at any accuracy, because it cannot
    separate two candidates from each other."""
    graded = [c for c in calls if c.parsed]
    verdicts = {c.verdict for c in graded}
    yes = [c for c in graded if c.supported]
    no = [c for c in graded if not c.supported]
    rate = lambda rows, want: (
        round(sum(1 for c in rows if c.verdict == want) / len(rows), 3)
        if rows else None)
    return {
        'n_items': len(calls),
        'n_parsed': len(graded),
        'schema_failures': len(calls) - len(graded),
        'accuracy': (round(sum(1 for c in graded
                               if c.verdict == int(c.supported)) / len(graded), 3)
                     if graded else None),
        # Per class: a constant predictor gives itself away as 1.0 and 0.0.
        'recall_supported': rate(yes, 1),
        'recall_unsupported': rate(no, 0),
        'degenerate': len(verdicts) < 2 and bool(graded),
        'seconds_per_call': (round(sum(c.seconds for c in calls) / len(calls), 2)
                             if calls else None),
        'prompt_tokens_max': max((c.usage.get('input_tokens') or 0
                                  for c in calls), default=0),
        'completion_tokens_mean': (
            round(sum(c.usage.get('output_tokens') or 0 for c in calls)
                  / len(calls), 1) if calls else None),
    }


def verdict_line(model: str, result: dict) -> str:
    stamp = ('DEGENERATE' if result['degenerate']
             else 'usable' if (result['accuracy'] or 0) >= 0.8
             and result['schema_failures'] == 0 else 'weak')
    return (f'{model:28} {stamp:11} acc={result["accuracy"]} '
            f'(+{result["recall_supported"]}/-{result["recall_unsupported"]}) '
            f'schema_fail={result["schema_failures"]}/{result["n_items"]} '
            f'{result["seconds_per_call"]}s per call, '
            f'{result["completion_tokens_mean"]} out tokens')


def screen(models: list[str], pairs: int = 6) -> dict:
    settings = load_lab_settings()
    if not settings.llm_ready:
        sys.exit('no LLM backend to screen: RAGLAB_LLM=claude and '
                 'RAGLAB_LLM=codex run a CLI already installed and logged in on '
                 'this machine and need no key at all; RAGLAB_LLM=ollama runs a '
                 'model on it; RAGLAB_LLM=openrouter needs OPENROUTER_API_KEY '
                 'for a remote candidate')
    diary = corpus.load_diary()
    sessions = corpus.sessions_by_id(diary)
    items = build_items(corpus.load_ground_truth(), sessions, pairs)
    signal = lexical_signal(items)
    print(f'{len(items)} items · {len(models)} models · via {settings.provider}')
    print(f'lexical signal: supported {signal["supported"]} vs unsupported '
          f'{signal["unsupported"]} overlap (difference {signal["difference"]}) — '
          f'{"word overlap cannot separate the classes" if signal["blind"] else
             "WARNING: the classes differ lexically, so this screen leaks"}\n')

    report = {'provider': settings.provider, 'pairs': pairs,
              'lexical_signal': signal,
              'items': [asdict(item) for item in items], 'models': {}}
    for i, model in enumerate(models, start=1):
        llm = judge_llm(settings, model)
        calls = []
        for j, item in enumerate(items, start=1):
            # One rewritten line per item, so a slow screen isn't silent.
            print(f'\r  [{i}/{len(models)}] {model[:28]:28} item {j}/{len(items)} '
                  f'{"supported" if item.supported else "fabricated":>10}'.ljust(78),
                  end='', flush=True)
            calls.append(ask(llm, item))
        print('\r'.ljust(80), end='')
        result = score(calls)
        report['models'][model] = {'score': result,
                                   'calls': [asdict(call) for call in calls]}
        print(verdict_line(model, result), flush=True)
    return report


def lexical_signal(items: list[Item]) -> dict:
    """How much a judge could score by counting words instead of reading. If the
    two classes' overlap with the context differs, the screen leaks; `blind` is
    the near-zero-difference property the screen needs."""
    mean = lambda rows: (round(sum(rows) / len(rows), 3) if rows else None)
    supported = mean([i.overlap for i in items if i.supported])
    unsupported = mean([i.overlap for i in items if not i.supported])
    if supported is None or unsupported is None:
        return {'supported': supported, 'unsupported': unsupported,
                'difference': None, 'blind': False}
    difference = round(supported - unsupported, 3)
    return {'supported': supported, 'unsupported': unsupported,
            'difference': difference, 'blind': abs(difference) <= 0.03}


def save(report: dict) -> Path:
    """Written to `.screens/`, never `.runs/` — a screen is not a leaderboard row."""
    SCREENS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    path = SCREENS_DIR / f'judgescreen-{stamp}.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                    encoding='utf-8')
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', required=True,
                        help='model ids to screen, e.g. qwen3.5:2b gemma4:e2b')
    parser.add_argument('--pairs', type=int, default=6,
                        help='supported/unsupported claim pairs per model '
                             '(default %(default)s, so %(default)s×2 calls)')
    args = parser.parse_args()
    report = screen(args.models, args.pairs)
    print(f'\nsaved {save(report)}')


if __name__ == '__main__':
    main()
