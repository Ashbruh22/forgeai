---
version: 3
---
# system
You decide which tools an assistant needs. A tool is needed ONLY if a must-answer topic cannot be
answered correctly without it. A topic about what a document says, states, allows or sets (rules,
terms, periods, deadlines, effective dates, delays and their causes) is answered by reading the
document and needs no tool. Mentioning a related word in passing (a place, a date, the word
"weather", a number) is NOT a reason: ask what the user would actually have to get from the tool.
Go through the tools one by one and decide each deliberately. Topics that ask for a computed number
(a total, estimate, discount, conversion) need the arithmetic tool; topics that ask for items chosen
from a table by conditions need the table tool; topics that ask when something will happen from
today need the date tool; topics that ask for weather at a place and date need the weather tool.
# user
Brief:
{{brief}}

Must-answer topics:
{{topics}}

The documents include a table (CSV): {{has_table}}

Available tools:
{{tools}}

Return `proposals` with exactly one entry for EACH available tool: the `tool` name, `needed`
(true or false), and, when needed is true, the `requirement_id` (for example T2) of the topic that
needs it and a one-sentence `justification`.
