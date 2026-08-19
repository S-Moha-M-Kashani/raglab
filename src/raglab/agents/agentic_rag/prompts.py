"""The five node prompts, one per model-calling node in the loop."""

PLAN_PROMPT = (
    'You plan retrieval over a personal Farsi diary. In one short sentence, say '
    'what evidence would answer the question — dates, names, events to look for. '
    'Do not answer the question itself.')
ASSESS_PROMPT = (
    'You judge whether retrieved diary excerpts are enough to answer a question. '
    'Reply with exactly one line: "SCORE: n" where n is 0-10 — 10 means the '
    'excerpts fully answer it, 0 means they are irrelevant. No other text.')
REWRITE_PROMPT = (
    'Rewrite a search query over a Farsi personal diary so it retrieves the '
    'missing evidence. Reply with the query only — keywords in Farsi, no '
    'explanation, no question words.')
CRITIQUE_PROMPT = (
    'You check a Farsi answer against the diary excerpts it was written from. '
    'Reply with exactly one line: "SCORE: n" where n is 0-10 — 10 means every '
    'claim in the answer is supported by the excerpts, 0 means it is invented. '
    'No other text.')
COMPLETENESS_PROMPT = (
    'You check whether a Farsi answer actually answers the question that was '
    'asked. Reply with exactly one line: "SCORE: n" where n is 0-10 — 10 means '
    'it answers it directly and completely, 0 means it does not answer it. No '
    'other text.')
