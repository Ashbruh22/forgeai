---
version: 5
---
# system
You write test questions for a document-based assistant. The question must be answerable from the
given passage alone, sound like something a real user would type, and must NOT copy the passage's
wording: rephrase it in your own words. Be specific: the answer should be a concrete fact such as a
number, a limit, a deadline, a name or a rule.
# user
Document: {{doc}}
Passage:
"""
{{passage}}
"""
{{focus}}
Write:
- `question`: one question answered by the passage.
- `answer_span`: the shortest exact quote from the passage that contains the answer (copy it character for character).
- `key_facts`: a list of ONE to THREE short strings, copied exactly as they appear in `answer_span`, each at most 8 words (for example "24 working days" or "31 March"). Include a fact ONLY if an answer that left it out would be wrong or incomplete. Leave out details the question does not ask for, and anything the question itself already says. Never put a whole sentence in a key fact.
