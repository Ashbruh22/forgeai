# Groundability gaps

## G-T4: unsupported_topic

The brief asks for 'Salary slip interpretation', but none of the 3 probe questions could be answered from the documents.

- recommended action: `refuse_with_redirect`
- redirect text: I do not have information on this. Please contact the relevant team directly.
- evidence: probe: How many annual leave days does a full-time shore employee accrue per month according to the salary slip? (span shares no content term with the topic); probe: Where is the information about an employee's leave accrual recorded, as mentioned in the salary slip policy? (span shares no content term with the topic); probe: What do the documents say about Salary slip interpretation? (model found no supporting span)

## Topic coverage

| topic | verdict | verified probes | evidence |
|---|---|---|---|
| Annual leave entitlements | SUPPORTED | 3/3 | leave_policy.pdf#c3, leave_policy.pdf#c4 |
| Travel expense reimbursement procedures | SUPPORTED | 3/3 | travel_expense_policy.pdf#c10, travel_expense_policy.pdf#c8 |
| Code of conduct rules | SUPPORTED | 3/3 | code_of_conduct_it_policy.pdf#c3, code_of_conduct_it_policy.pdf#c6 |
| Salary slip interpretation | UNSUPPORTED | 0/3 | - |

## Retrieval threshold calibration

- `min_score` = 0.6571 (bounds [0.5571, 0.7571])
- LOW CONFIDENCE: distributions overlap (in-scope p10=0.707, near-miss p90=0.817); threshold set 0.05 below the in-scope p10 to preserve recall; unanswerable near-misses rely on the refusal policy
