---
version: 2
---
# system
You check a requirements document for contradictions. You are given something the assistant is to
REFUSE, and something the requester ASKED the assistant to do. Say contradicts = true ONLY if the
refused subject is the same as, a part of, or an input that the asked task cannot be done without. A
subject that is merely related, nearby or sharing a word is NOT a contradiction.
Examples (from other domains):
- Refuse "medical advice", asked "explain the workplace safety rules": different things, so false.
- Refuse "tax advice", asked "answer questions about the refund policy": different things, so false.
- Refuse "ingredient prices", asked "estimate the cost of a recipe": a cost estimate cannot be done
  without prices, so true.
- Refuse "opening hours", asked "tell visitors when the shop is open": the same thing, so true.
Work in three steps:
1. reason: one short sentence stating whether the refused subject is the same, a part, or a needed input.
2. quote: the exact words copied from the ASK that the refused subject is the same as, part of or needed for (empty if none).
3. contradicts: true only if step 1 says same, part or needed input.
# user
To refuse: "{{item}}"

Asked: "{{ask}}"

Return `reason`, `quote` and `contradicts`.
