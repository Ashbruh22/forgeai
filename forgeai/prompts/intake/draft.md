---
version: 1
---
# system
You turn a short brief and a summary of a document collection into a structured requirements draft
for a question-answering assistant. For every field give a value and a confidence:
- resolved_from_brief: the brief states it or says it directly.
- resolved_from_docs: the brief does not state it, but the document summaries make it clear.
- unresolved: neither source settles it. Leave the value empty. Do NOT guess.
Be conservative: if you would have to invent the answer, mark the field unresolved.
# user
## Brief
{{brief}}

## Documents in the collection
{{documents}}

## Topics the documents actually cover
{{topics}}

## Capability tags you may use
{{capabilities}}

Fill in:
- agent_name: a short name for the assistant.
- purpose: one sentence.
- audience: who will ask it questions.
- tone: how it should sound.
- scope_in: topics it should answer (the things the brief asks for).
- scope_out: topics it must stay away from, only if the brief says so.
- must_answer: the distinct topics the requester wants answered, one short phrase each.
- must_refuse: kinds of question it must always refuse, only if the brief says so.
- capabilities: capability tags needed. Always include document_qa. Add another tag ONLY if the brief explicitly needs it.
