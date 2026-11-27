---
version: 1
---
# system
You design calculation questions whose answers are computed by code from facts in the documents. You do
NOT compute the answer. You choose the facts and write the formula; every fact must be copied exactly.
Variables come from three places: a table row (source "csv"), a number inside a quoted passage (source
"prose") or a number you put in the question itself (source "question"), such as a quantity.
# user
Table rows (one is a candidate):
{{rows}}

Passages that may hold rules (discounts, tiers, rates, limits):
{{passages}}

Design one question that needs at least one table value (or a passage number) and one passage number.
Return:
- `question`: the question for a user. It must contain the number of any "question" variable.
- `variables`: each with `name` (a short identifier such as price), `source` (csv, prose or question),
  and for csv: `doc`, `row_key` (the column/value pairs that identify exactly one row) and `column`;
  for prose: `doc`, `span_text` (exact quote from a passage) and `number_text` (the number as written in it);
  for question: `number_text`.
- `formula`: an arithmetic expression over the variable names, written so that a percentage such as 5%
  is entered as 5 and divided by 100 in the formula.
- `conditions`: comparisons over the variables that must be true for the formula to apply (for example
  that an order total falls inside the tier whose discount you used). Use an empty list if none.
