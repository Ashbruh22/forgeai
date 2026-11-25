---
version: 1
---
# system
You label clusters of related text passages. Be specific and concrete. Use only the passages shown.
# user
These passages were grouped together because they discuss related things:

{{samples}}

Return a `label` (2 to 6 words naming the topic) and a `description` (one sentence saying what
questions these passages could answer).
