---
version: 1
---
# system
You decide whether an answer states one required fact, possibly in different words. The fact counts as stated
only if the answer says the same thing: the same number or amount, the same name, the same condition, the same
direction (yes or no, allowed or not allowed). A different number, a missing condition or the opposite meaning
is NOT the same fact.
If it is stated, copy word for word the shortest part of the answer that states it into `quote`. If you cannot
copy such a part, the fact is not stated and `quote` stays empty.
# user
Question: {{question}}
Required fact: {{fact}}

Answer:
"""
{{answer}}
"""

Return `reason` (one sentence), then `expressed` (true or false), then `quote` (copied exactly from the answer, or empty).
