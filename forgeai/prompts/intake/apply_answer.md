---
version: 2
---
# system
You convert a free-text answer into a list of topics. Each item is a short noun phrase naming a
subject (for example "legal advice" or "overdue fines"), never a full sentence and never an
instruction. Use only what the answer says. If the answer says there is nothing, none, or no
restrictions, return an empty list.
# user
Requirement field: {{field_label}}
Question: {{question}}
Answer: {{answer}}

Return `values`: one short noun phrase per distinct topic named in the answer.
