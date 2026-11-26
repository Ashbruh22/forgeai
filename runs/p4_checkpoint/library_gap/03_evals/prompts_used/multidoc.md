---
version: 4
---
# system
You write test questions that need TWO documents. The question must not be answerable from either
passage alone, must sound like a real user's question, and must not copy the passages' wording.
# user
Passage A (from {{doc_a}}):
"""
{{passage_a}}
"""

Passage B (from {{doc_b}}):
"""
{{passage_b}}
"""

Write one `question` that needs a fact from A and a fact from B. Then give `span_a` and `span_b`, each the shortest exact quote (copied character for character) from its own passage holding the needed fact, and `facts_a` and `facts_b`: lists of one or two short strings (only what the question asks for) (at most 8 words each, never a whole sentence), copied exactly from the respective span.
