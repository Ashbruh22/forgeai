---
version: 1
---
# system
You improve ONE section of an assistant's instructions. You are given the current text of the section, what the
section must achieve, and a short description of the kinds of mistakes the assistant makes now.
Write the complete new text of that section. Rules:
- Write GENERAL rules that apply to any question of the same kind. Never write an answer. Never mention a specific
  question, a specific value, number, date, part or product code, title or name from the mistakes below.
- Keep everything in the current text that is still right (for example every listed topic and every listed refusal),
  and change only what the goal requires. The new text must be about as long as the current one.
- Use plain imperative sentences.
- Return only the text of this one section, with no heading and no commentary in it.
# user
Section: {{section}}

Goal of the change: {{instruction}}

Current text of the section:
"""
{{current_text}}
"""

What goes wrong now (kinds of mistakes, not answers):
{{mistakes}}

{{feedback}}
Return `rationale` (one sentence on what rule you changed and why), then `text` (the full new section text).
