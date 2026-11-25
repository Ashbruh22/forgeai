# Groundability gaps

## G-T2: partial_coverage

Only part of 'Supplier comparisons for spares' is supported by the documents (1 of 3 probes verified).

- recommended action: `warn`
- evidence: terms_kestrel_marine_spares.pdf#c2

## G-LIVE-1: live_data

The brief asks for current or real-time information, but the documents are a static snapshot and no tool provides it: "It should also tell them what's in stock right now at each port." The recorded version (Recorded stock port location) IS answerable from the documents as of 2026-11-01; only the live value is refused.

- recommended action: `refuse_with_redirect`
- unsupported part: Current stock availability at port
- redirect text: I do not have information on this. Please contact the relevant team directly.
- evidence: It should also tell them what's in stock right now at each port.

## Topic coverage

| topic | verdict | verified probes | supported part | evidence |
|---|---|---|---|---|
| Current stock levels at ports | PARTIAL | 2/3 | Recorded stock port location | spares_catalogue.csv#c208, spares_catalogue.csv#c209 |
| Supplier comparisons for spares | PARTIAL | 1/3 | Availability status of ship spares from suppliers | terms_kestrel_marine_spares.pdf#c2 |
| Order cost estimates with discounts | SUPPORTED | 3/3 | - | terms_norvik_marine_supply.pdf#c3, terms_kestrel_marine_spares.pdf#c3 |
| Estimated arrival times for parts | SUPPORTED | 3/3 | - | spares_catalogue.csv#c125, spares_catalogue.csv#c137, spares_catalogue.csv#c124 |

## Agreement between the two support checks

- probes with a verbatim span: 21 of 32
- both pass: 14; code check only: 7; judge only: 0; neither: 0
- disagreement rate: 0.333

## Retrieval threshold calibration

- `min_score` = 0.6573 (bounds [0.5573, 0.7573])
- LOW CONFIDENCE: distributions overlap (in-scope p10=0.707, near-miss p90=0.805); threshold set 0.05 below the in-scope p10 to preserve recall; unanswerable near-misses rely on the refusal policy
