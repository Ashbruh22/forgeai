---
version: 2
---
# system
You write questions that sound closely related to a topic but ask for something a typical document
on that topic would NOT contain: a neighbouring subject, a more specific personal case, or a
different organisation's rules. They must still read like real user questions.
# user
Assistant purpose: {{purpose}}
Topic: {{topic}}

Document excerpts on this topic (what the documents DO contain):
{{samples}}

Write 2 near-miss questions for this topic: related, but not answerable from the excerpts above.
Return them as `questions`.
