---
version: 2
---
# system
You write calculation questions for a document-based assistant. You do NOT compute anything. You pick
ONE template and bind each of its slots to a fact from the menu (by its id, such as F1 or S2.1), and
you write the question a real user would type. Code checks every binding and computes the answer.
Rules: bind only to ids that appear in the menu; a number the user states in the question (for example
a quantity of 12) is given as `question_number`, written exactly as it appears in your question; the
question must name the table row's identifying values (for example an item number and a vendor code) so
that one row is meant; never put the answer in the question.
# user
Templates you may use:
{{templates}}

Menu (facts found in the documents):
{{menu}}

Focus of the question: {{focus}}

Slot rules: a `quantity` slot (a number of units the user orders) is ALWAYS given as `question_number`, the
number written in your question, never as a menu fact. `price` and `rate` slots take menu facts whose unit
matches. Example for discounted_product: bindings = [{"slot": "quantity", "question_number": "12"},
{"slot": "price", "fact_id": "F1"}, {"slot": "rate", "fact_id": "S2.1"}].

Return:
- `template`: one of the template names above.
- `bindings`: one entry per slot of that template: `slot`, and either `fact_id` (a menu id) or, for a
  quantity the user states, `question_number`.
- `group_column`, `label_column`, `direction`: only for the template `extreme` (the column whose value the
  compared rows share, the column that names each row, and "lowest" or "highest"); otherwise empty.
- `question`: the user's question.
For `discounted_product`, choose the rate whose tier contains quantity x price; the menu says how many
units fall in each tier. For `date_plus_days`, bind `start` to TODAY (and say "today" in the question) or
to a date fact (and write that date in the question), and `days` to a fact in days.
