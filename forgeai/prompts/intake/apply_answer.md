---
version: 1
---
# system
You convert a free-text answer into the list of short items it contains. Use only what the answer says.
# user
Requirement field: {{field_label}}
Question: {{question}}
Answer: {{answer}}

Return `values`: one short phrase per distinct item in the answer (a single item if the answer is a single statement).
