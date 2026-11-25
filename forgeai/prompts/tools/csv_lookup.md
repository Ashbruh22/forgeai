---
version: 1
---
# user
csv_lookup: filters rows of a CSV file in the document folder and returns the matching rows (at most 20).
args: {"file": "<csv file name>", "filters": {"<column>": <value or {"op": "=|!=|<|<=|>|>=|contains", "value": <value>}>}, "columns": ["<column>", ...]}. "filters" and "columns" are optional. Text comparison ignores case. Example: {"file": "parts.csv", "filters": {"equipment": "Main Engine", "unit_price_usd": {"op": "<", "value": 500}}, "columns": ["part_no", "unit_price_usd"]}
