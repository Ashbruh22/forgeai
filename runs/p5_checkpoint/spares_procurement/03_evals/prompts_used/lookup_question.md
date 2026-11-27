---
version: 1
---
# system
You turn a table lookup into a natural question. Mention the identifying values exactly as given and
never reveal the value being asked for.
# user
Table: {{table}}
Identifying values: {{key}}
Column asked for: {{column}}

Write one natural `question` asking for the value of that column for the item with those identifying values.
