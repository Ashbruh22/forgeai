---
version: 1
---
# system
You write under-specified questions. Two passages state the same kind of fact with different values
for different situations. Write ONE question a real user might ask about that kind of fact that does
not say which situation it means, so that it has two valid readings. The question must not contain any
of the words that tell the situations apart, and must still name the kind of fact being asked about.
# user
Reading 1 ({{label_a}}): "{{sentence_a}}"
Reading 2 ({{label_b}}): "{{sentence_b}}"

Words that tell the readings apart, which the question must NOT contain: {{discriminators}}

Return:
- `question`: the under-specified question.
- `reading_a`: a short phrase for what the user would mean under reading 1.
- `reading_b`: a short phrase for what the user would mean under reading 2.
