# Groundability gaps

## G-LIVE-1: live_data

The brief asks for current or real-time information, but the documents are a static snapshot and no tool provides it: "It should also tell them what's in stock right now at each port." The recorded version (Where is the item listed as being held (stock port) according to the documents?) IS answerable from the documents as of 2026-11-01; only the live value is refused.

- recommended action: `refuse_with_redirect`
- unsupported part: What is the current availability status of the item at each port?
- redirect text: I do not have information on this. Please contact the relevant team directly.
- evidence: It should also tell them what's in stock right now at each port.

## Topic coverage

| topic | verdict | verified probes | supported part | evidence |
|---|---|---|---|---|
| What is in stock at each port? | PARTIAL | 3/3 | Where is the item listed as being held (stock port) according to the documents? | spares_catalogue.csv#c208, spares_catalogue.csv#c210 |
| Which suppliers offer the part? | SUPPORTED | 3/3 | - | spares_catalogue.csv#c173, spares_catalogue.csv#c172 |
| What is the cost with discounts? | SUPPORTED | 3/3 | - | terms_norvik_marine_supply.pdf#c3, terms_kestrel_marine_spares.pdf#c3 |
| When will the parts arrive? | SUPPORTED | 3/3 | - | spares_catalogue.csv#c163, spares_catalogue.csv#c164 |

## Agreement between the two support checks

- probes with a verbatim span: 21 of 23
- both pass: 17; code check only: 4; judge only: 0; neither: 0
- disagreement rate: 0.19

## Retrieval threshold calibration

- `min_score` = 0.6706 (bounds [0.5706, 0.7706])
- LOW CONFIDENCE: distributions overlap (in-scope p10=0.721, near-miss p90=0.781); threshold set 0.05 below the in-scope p10 to preserve recall; unanswerable near-misses rely on the refusal policy
