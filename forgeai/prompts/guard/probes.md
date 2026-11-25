---
version: 2
---
# system
You write realistic questions that a user of an assistant might ask. Each question is self-contained
and specific, the kind with a concrete answer such as a number, a rule, a deadline, a name or a
value in a table. Use the same names, codes and terms that appear in the document excerpts when the
excerpts cover the topic. If the excerpts do not mention the topic, still ask about the topic itself.
# user
Assistant purpose: {{purpose}}
Topic: {{topic}}

Document excerpts that may mention this topic:
{{samples}}

Write 3 different questions a real user might ask about this topic. Every question must clearly be
about the topic. Return them as `questions`.
