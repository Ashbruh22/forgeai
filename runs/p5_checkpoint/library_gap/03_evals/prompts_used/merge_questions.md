---
version: 1
---
# system
You combine two questions about different documents into ONE natural question that needs both
documents to answer. Keep the specifics of each question. Do not answer it and do not copy either
question word for word.
# user
Question about {{doc_a}}: {{question_a}}
Question about {{doc_b}}: {{question_b}}

Write one question a real user could ask that needs the answer to both. Return it as `question`.
