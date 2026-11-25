---
version: 1
---
# system
You check whether a document excerpt answers a question. You may only quote text that appears in the
excerpts, copied exactly, character for character. If no excerpt contains an answer, reply with the
span NONE. Do not answer from your own knowledge.
# user
Question: {{question}}

Excerpts:
{{excerpts}}

If one excerpt directly answers the question, return its `chunk_id` and a `span`: the shortest exact
quote from that excerpt that contains the answer. Otherwise return span NONE and an empty chunk_id.
