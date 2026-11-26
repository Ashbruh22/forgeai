---
version: 4
---
# system
You write test questions that are genuinely ambiguous. Given passages that treat related things
differently, write ONE question that a real user might ask, which could mean either reading and does
NOT say which. The deciding words (the discriminators) must be absent from the question.
# user
Reading 1 passage (from {{doc_a}}):
"""
{{passage_a}}
"""

Reading 2 passage (from {{doc_b}}):
"""
{{passage_b}}
"""

Return `question` (it must not contain any discriminator) and `readings`: exactly two entries, each with
`reading` (what the user would mean), `discriminators` (one or two words that would select this reading,
for example "annual" or "sick"), `span` (the shortest exact quote from that passage that answers the
question under this reading) and `key_facts` (a list of one or two strings of at most 8 words each, copied exactly from the span; never a whole sentence).
