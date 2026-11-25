# Groundability gaps

## G-T5: unsupported_topic

The brief asks for 'What are the cafe menu options?', but none of the 3 probe questions could be answered from the documents.

- recommended action: `refuse_with_redirect`
- redirect text: I do not have information on this. Please contact the relevant team directly.
- evidence: probe: What are the current menu options available at the Brightwater Library cafe? (model found no supporting span); probe: Can you list the different items served on the Brightwater Library cafe menu? (model found no supporting span); probe: What food and drink options are available at the Brightwater Library cafe? (model found no supporting span)

## G-LIVE-1: live_data

The brief asks for current or real-time information, but the documents are a static snapshot and no tool provides it: 'It should also tell visitors whether a specific book is on the shelf right now' The recorded version (Recorded shelf location and copy count) IS answerable from the documents as of 2026-11-20; only the live value is refused.

- recommended action: `refuse_with_redirect`
- unsupported part: Current availability of the book on the shelf
- redirect text: I do not have information on this. Please contact the relevant team directly.
- evidence: It should also tell visitors whether a specific book is on the shelf right now

## Topic coverage

| topic | verdict | verified probes | supported part | evidence |
|---|---|---|---|---|
| Are there any fines for overdue books? | SUPPORTED | 3/3 | - | library_rules.md#c1 |
| What is the loan period for books? | SUPPORTED | 2/3 | - | library_rules.md#c0 |
| Can I renew my library card? | PARTIAL | 1/3 | Renewal eligibility for library card members | library_rules.md#c0 |
| Is the book 'Pride and Prejudice' available? | SUPPORTED | 3/3 | - | books.csv#c2 |
| What are the cafe menu options? | UNSUPPORTED | 0/3 | - | - |
| tell visitors whether a specific book is on the shelf right now | PARTIAL | 2/3 | Recorded shelf location and copy count | books.csv#c7, books.csv#c4 |

## Agreement between the two support checks

- probes with a verbatim span: 27 of 42
- both pass: 21; code check only: 5; judge only: 0; neither: 1
- disagreement rate: 0.185

## Retrieval threshold calibration

- `min_score` = 0.706 (bounds [0.606, 0.806])
- LOW CONFIDENCE: distributions overlap (in-scope p10=0.756, near-miss p90=0.807); threshold set 0.05 below the in-scope p10 to preserve recall; unanswerable near-misses rely on the refusal policy
