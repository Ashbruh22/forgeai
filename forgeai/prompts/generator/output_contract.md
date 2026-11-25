---
version: 2
---
# user
Reply with a single JSON object that follows the output contract (contracts/answer_v1.json). Use status "answered" with citations for answers, "refused" when you decline, and "needs_clarification" with a clarifying_question when the question is ambiguous. The "answer" field must never be empty, including when you refuse: say briefly that you cannot help and give the redirect. Never add text outside the JSON.
