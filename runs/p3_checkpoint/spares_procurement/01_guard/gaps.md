# Groundability gaps

## G-T1: unsupported_topic

The brief asks for 'What spares are in stock at each port?', but none of the 3 probe questions could be answered from the documents.

- recommended action: `refuse_with_redirect`
- redirect text: I do not have information on this. Please contact the relevant team directly.
- evidence: probe: Which ship spares are currently in stock at the port of Calder Bay according to the supplier terms? (model found no supporting span); probe: What is the list of spares available at each port, as provided by the suppliers S02 and S03? (model found no supporting span); probe: Which specific spares are available at the ports mentioned in the supplier terms for Calder Bay Ship Management Ltd? (model found no supporting span)

## G-LIVE-1: live_data

The brief asks for current or real-time information, but the documents are a static snapshot and no tool provides it: "It should also tell them what's in stock right now at each port."

- recommended action: `refuse_with_redirect`
- evidence: It should also tell them what's in stock right now at each port.

## Topic coverage

| topic | verdict | verified probes | evidence |
|---|---|---|---|
| What spares are in stock at each port? | UNSUPPORTED | 0/3 | - |
| Which suppliers offer the best price for a given part? | SUPPORTED | 3/3 | spares_catalogue.csv#c46, spares_catalogue.csv#c173, spares_catalogue.csv#c50 |
| What is the estimated cost of an order with discounts? | SUPPORTED | 2/3 | terms_norvik_marine_supply.pdf#c3, terms_kestrel_marine_spares.pdf#c3 |
| When will a part arrive after ordering? | SUPPORTED | 2/3 | spares_catalogue.csv#c173 |

## Retrieval threshold calibration

- `min_score` = 0.6548 (bounds [0.5548, 0.7548])
- LOW CONFIDENCE: distributions overlap (in-scope p10=0.705, near-miss p90=0.842); threshold set 0.05 below the in-scope p10 to preserve recall; unanswerable near-misses rely on the refusal policy
