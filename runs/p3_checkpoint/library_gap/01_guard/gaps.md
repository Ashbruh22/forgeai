# Groundability gaps

## G-T4: unsupported_topic

The brief asks for 'Is a specific book available on the shelf?', but none of the 3 probe questions could be answered from the documents.

- recommended action: `refuse_with_redirect`
- redirect text: I do not have information on this. Please contact the relevant team directly.
- evidence: probe: Is the book 'The Great Gatsby' available on the shelf? (model found no supporting span); probe: Can I check the availability of the book 'To Kill a Mockingbird'? (model found no supporting span); probe: Is the book '1984' currently on the shelf at Brightwater Library? (model found no supporting span)

## G-T5: unsupported_topic

The brief asks for 'What is the cafe menu?', but none of the 3 probe questions could be answered from the documents.

- recommended action: `refuse_with_redirect`
- redirect text: I do not have information on this. Please contact the relevant team directly.
- evidence: probe: What are the available items on the Brightwater Library cafe menu? (model found no supporting span); probe: Is there a cafe menu available at the Brightwater Library? (model found no supporting span); probe: What food and drink options are offered at the Brightwater Library cafe? (model found no supporting span)

## G-LIVE-1: live_data

The brief asks for current or real-time information, but the documents are a static snapshot and no tool provides it: 'It should also tell visitors whether a specific book is on the shelf right now, and answer questions about the cafe menu.'

- recommended action: `refuse_with_redirect`
- evidence: It should also tell visitors whether a specific book is on the shelf right now, and answer questions about the cafe menu.

## Topic coverage

| topic | verdict | verified probes | evidence |
|---|---|---|---|
| What is the loan period for books? | SUPPORTED | 3/3 | library_rules.md#c0 |
| What are the fines for overdue items? | SUPPORTED | 3/3 | library_rules.md#c1 |
| What are the requirements for a library card? | SUPPORTED | 3/3 | library_rules.md#c2, library_rules.md#c0 |
| Is a specific book available on the shelf? | UNSUPPORTED | 0/3 | - |
| What is the cafe menu? | UNSUPPORTED | 0/3 | - |

## Retrieval threshold calibration

- `min_score` = 0.6627 (bounds [0.5627, 0.7627])
- LOW CONFIDENCE: distributions overlap (in-scope p10=0.713, near-miss p90=0.779); threshold set 0.05 below the in-scope p10 to preserve recall; unanswerable near-misses rely on the refusal policy
