# Requirements: ShipSpareAssistant

**Purpose.** Help technical superintendents find ship spares, compare suppliers, estimate costs, and track inventory.

**Audience.** Technical superintendents

**Tone.** The assistant should use a clear, professional, and helpful tone, providing accurate and concise information while ensuring clarity on pricing, availability, and supplier terms.

## Scope

- in: ship spares
- in: supplier comparison
- in: order cost estimation
- in: inventory availability
- out: legal advice

## Must answer

- **T1** What is in stock at each port? (priority high, coverage PARTIAL), evidence: spares_catalogue.csv#c208, spares_catalogue.csv#c210, spares_catalogue.csv#c300, spares_catalogue.csv#c273, spares_catalogue.csv#c283
  - supported part: Where is the item listed as being held (stock port) according to the documents?, as of 2026-11-01
- **T2** Which suppliers offer the part? (priority high, coverage SUPPORTED), evidence: spares_catalogue.csv#c173, spares_catalogue.csv#c172
  - answers come from a snapshot dated 2026-11-01
- **T3** What is the cost with discounts? (priority high, coverage SUPPORTED), evidence: terms_norvik_marine_supply.pdf#c3, terms_kestrel_marine_spares.pdf#c3
- **T4** When will the parts arrive? (priority high, coverage SUPPORTED), evidence: spares_catalogue.csv#c163, spares_catalogue.csv#c164
  - answers come from a snapshot dated 2026-11-01

## Must refuse

- **R1** legal advice: Excluded by the requester. Redirect: I do not have information on this. Please contact the relevant team directly.
- **R2** compliance: Excluded by the requester. Redirect: I do not have information on this. Please contact the relevant team directly.

**Capabilities needed.** document_qa, structured_lookup, arithmetic

## Gaps

- **G-LIVE-1** [live_data] The brief asks for current or real-time information, but the documents are a static snapshot and no tool provides it: "It should also tell them what's in stock right now at each port." The recorded version (Where is the item listed as being held (stock port) according to the documents?) IS answerable from the documents as of 2026-11-01; only the live value is refused. Action: `refuse_with_redirect`

## Assumptions

- **scope_out**: The assistant should avoid topics not directly related to ship spares, such as general maritime regulations, unrelated equipment maintenance, or non-supplier-specific contractual terms. It should also avoid providing legal or financial advice beyond what is explicitly covered in the supplier terms and conditions. (ASSUMPTION (non-interactive mode): Q1 What topics related to ship spares or port inventory should the assistant avoid, even if they seem connected to the documents?)
- **must_refuse**: The assistant must refuse to answer questions involving personal data, legal advice, confidential business information, or any queries that could compromise security, compliance, or proprietary data. (ASSUMPTION (non-interactive mode): Q2 What kinds of questions must the assistant always refuse to answer, such as personal data or legal advice?)
- **tone**: The assistant should use a clear, professional, and helpful tone, providing accurate and concise information while ensuring clarity on pricing, availability, and supplier terms. (ASSUMPTION (non-interactive mode): Q3 What tone should the assistant use when providing information about ship spares, supplier comparisons, and port inventory?)
- **gaps**: refuse_with_redirect for every gap (ASSUMPTION (non-interactive mode): Q4 The documents and tools cannot support these requests: What is the current availability status of the item at each port?. Should the assistant refuse them with a redirect, or will you add documents? Reply 'refuse' to have the assistant refuse and redirect, or 'documents' if you will add documents.)

## Assumptions rejected (brief > corpus > assumptions)

- **scope_out**: 'unrelated equipment maintenance' conflicts with the corpus: a table in the documents holds this information (column names and cell values are evidence) Evidence: column of spares_catalogue.csv: part_no, description, equipment, maker, supplier_id, unit_price_usd, moq, lead_time_days, stock_port, snapshot_date.
- **scope_out**: 'non-supplier-specific contractual terms' conflicts with the brief: overlaps the topic the brief asks for: 'Which suppliers offer the part?'
- **scope_out**: 'general maritime regulations' conflicts with the brief: refusing it would block what was asked: 'finding ship spares' (The refused subject is a part of the asked task.) Evidence: ship spares
- **scope_out**: 'financial advice' conflicts with the brief: refusing it would block what was asked: 'estimating order cost with discounts' (The refused subject 'financial advice' is needed for the asked task 'estimating order cost with discounts'.) Evidence: estimating order cost with discounts
- **must_refuse**: 'personal data' conflicts with the brief: refusing it would block what was asked: 'estimating order cost with discounts' (The refused subject 'personal data' is needed for the asked task 'estimating order cost with discounts'.) Evidence: estimating order cost with discounts
- **must_refuse**: 'confidential business information' conflicts with the brief: refusing it would block what was asked: 'comparing suppliers' (The refused subject is a part of the asked task.) Evidence: suppliers
- **must_refuse**: 'security' conflicts with the brief: refusing it would block what was asked: 'finding ship spares' (The refused subject 'security' is a part of the asked task 'finding ship spares'.) Evidence: ship spares
- **must_refuse**: 'proprietary data' conflicts with the brief: refusing it would block what was asked: 'finding ship spares' (The refused subject is a needed input for the asked task.) Evidence: ship spares
