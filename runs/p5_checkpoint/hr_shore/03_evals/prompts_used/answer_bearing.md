---
version: 1
---
# system
You check one test fact. A correct answer to the question is compared with a list of key facts. A key fact
belongs in the list only if an answer that left it out would be WRONG or INCOMPLETE for this question.
A fact that merely adds detail, repeats something the question already says, or answers a different
question than the one asked does not belong.
Examples:
- Question "How many days do I have to submit my claim?" fact "30 days": needed, it is the answer.
- Question "How many days do I have to submit my claim?" fact "from the date of return": detail, the
  question does not ask when the clock starts.
- Question "Who approves leave over 10 days?" fact "the department head": needed, it is the answer.
# user
Question: {{question}}
Key fact: {{fact}}
Passage it comes from: "{{span}}"

Return `reason` (one sentence), then `needed` (true only if an answer without this fact would be wrong or incomplete).
