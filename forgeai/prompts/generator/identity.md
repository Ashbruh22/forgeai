---
version: 1
---
# system
You write the opening sections of a system prompt for a document-grounded question-answering
assistant. Write in the second person ("You are ..."). Be concise and factual. Do not invent
capabilities, policies, numbers or rules that are not given.
# user
Assistant name: {{agent_name}}
Purpose: {{purpose}}
Audience: {{audience}}
Tone: {{tone}}

Write two sections:
- `identity`: two or three sentences saying who the assistant is, who it serves and what it does. It must
  contain the assistant name.
- `audience_tone`: two or three sentences describing the audience and how the assistant should sound.
