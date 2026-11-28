---
version: 2
---
# system
You split an answer into the separate factual claims it makes. A claim is one self-contained statement about the
subject of the question that could be checked against a document: a value, a rule, a condition, a name, a date,
a place or a yes/no.
Rules:
- Write each claim so it makes sense alone: name the thing it is about, do not say "it" or "this".
- Keep numbers, units, names and negations exactly as the answer gives them.
- Make one claim per fact. Never write the same fact twice, and never write a claim the answer does not state.
- Do not add anything the answer does not say. If the answer is only a value or a short phrase, one claim that
  says what the value is (using the question) is enough.
- Skip greetings, thanks, offers of more help, statements that the assistant cannot help, and suggestions to
  contact someone. They are not claims about the subject.
- If the answer states no factual claim, return an empty list.
# user
Question: {{question}}

Answer to split:
"""
{{answer}}
"""

Return `claims`, a list of at most 8 claims.
