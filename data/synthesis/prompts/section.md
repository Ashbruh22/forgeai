---
version: 2
---
# system
You write sections of internal corporate documents for a FICTIONAL company, as numbered clauses.
Use formal, plain business English. Each clause is one to three sentences. Do not use markdown,
bullet points or clause numbers inside the clauses. Never mention real companies or real people.
# user
Company: {{company}}
Document: {{document}}
Section: {{section}}

What this section covers:
{{brief}}

Required facts. Every item below must be stated in the clauses, and the quoted text must be copied
exactly as written:
{{facts}}

Write {{n_clauses}} clauses, and never fewer than {{min_clauses}}. Elaborate with practical detail (who does what, when, and how) but add no new numbers. Every clause is a separate string in the list.

Hard rules:
- Do not state any number, amount, percentage or time limit other than the ones inside the required facts above.
- Do not mention any of these words: {{forbidden}}.
- Do not refer to other sections by number.

Return the `clauses` as a list of strings.
