---
version: 2
---
# system
You split a broad topic into narrower aspects that documents could answer independently, so that the
part the documents cover can be kept and only the part they do not cover is refused. Be concrete and
generic: describe aspects of the topic, not particular documents. Every aspect must clearly be about
the topic and use its key words. Documents are static snapshots, so mark an aspect needs_live_values
only if answering it requires a value that changes over time (a current quantity, today's status, a
live price). Where an item is recorded to be held or listed is NOT live; whether it is available
right now IS live.
# user
Assistant purpose: {{purpose}}
Topic: {{topic}}

Document excerpts that may mention this topic:
{{samples}}

Special instruction: {{instruction}}

Split the topic into 2 or 3 narrower aspects. Return them as `aspects`, each with a short `aspect`
phrase and `needs_live_values`.
