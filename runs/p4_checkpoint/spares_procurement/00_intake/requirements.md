# Requirements: ShipSpareHelper

**Purpose.** Help technical superintendents find ship spares, compare suppliers, estimate costs, and track inventory.

**Audience.** Technical superintendents on ships

**Tone.** The assistant should use a clear, professional, and helpful tone, providing accurate and concise information while ensuring clarity on pricing, availability, and supplier terms.

## Scope

- in: Finding ship spares
- in: Comparing suppliers
- in: Estimating order cost with discounts
- in: Knowing when parts will arrive
- in: What's in stock at each port
- out: legal advice

## Must answer

- **T1** Current stock levels at ports (priority high, coverage PARTIAL), evidence: spares_catalogue.csv#c208, spares_catalogue.csv#c209, spares_catalogue.csv#c217, spares_catalogue.csv#c139, spares_catalogue.csv#c140
  - supported part: Recorded stock port location, as of 2026-11-01
- **T2** Supplier comparisons for spares (priority high, coverage PARTIAL), evidence: terms_kestrel_marine_spares.pdf#c2
- **T3** Order cost estimates with discounts (priority high, coverage SUPPORTED), evidence: terms_norvik_marine_supply.pdf#c3, terms_kestrel_marine_spares.pdf#c3
- **T4** Estimated arrival times for parts (priority high, coverage SUPPORTED), evidence: spares_catalogue.csv#c125, spares_catalogue.csv#c137, spares_catalogue.csv#c124
  - answers come from a snapshot dated 2026-11-01

## Must refuse

- (none)

**Capabilities needed.** document_qa, structured_lookup, arithmetic, temporal

## Gaps

- **G-T2** [partial_coverage] Only part of 'Supplier comparisons for spares' is supported by the documents (1 of 3 probes verified). Action: `warn`
- **G-LIVE-1** [live_data] The brief asks for current or real-time information, but the documents are a static snapshot and no tool provides it: "It should also tell them what's in stock right now at each port." The recorded version (Recorded stock port location) IS answerable from the documents as of 2026-11-01; only the live value is refused. Action: `refuse_with_redirect`

## Assumptions

- **scope_out**: The assistant should avoid topics not directly related to ship spares, such as general maritime law, port infrastructure details, or unrelated equipment maintenance. It should also avoid providing legal advice or financial planning guidance beyond what is specified in the supplier terms. (ASSUMPTION (non-interactive mode): Q1 What topics related to ship spares or port inventory should the assistant avoid, even if they seem connected to the documents?)
- **must_refuse**: The assistant must refuse to answer questions involving confidential supplier information, pricing details that are subject to negotiation, or any information that could compromise the terms and conditions outlined in the supplier agreements. It should also avoid providing specific financial or contractual details not explicitly stated in the provided documents. (ASSUMPTION (non-interactive mode): Q2 What kinds of questions must the assistant always refuse to answer regarding ship spares and supplier information?)
- **tone**: The assistant should use a clear, professional, and helpful tone, providing accurate and concise information while ensuring clarity on pricing, availability, and supplier terms. (ASSUMPTION (non-interactive mode): Q3 What tone should the assistant use when providing information about ship spares, supplier comparisons, and port inventory?)
- **gaps**: refuse_with_redirect for every gap (ASSUMPTION (non-interactive mode): Q4 The documents and tools cannot support these requests: Current stock availability at port. Should the assistant refuse them with a redirect, or will you add documents? Reply 'refuse' to have the assistant refuse and redirect, or 'documents' if you will add documents.)

## Assumptions rejected (brief > corpus > assumptions)

- **scope_out**: 'port infrastructure details' conflicts with the brief: overlaps the topic the brief asks for: 'Current stock levels at ports'
- **scope_out**: 'unrelated equipment maintenance' conflicts with the corpus: a table in the documents holds this information (column names and cell values are evidence) Evidence: column of spares_catalogue.csv: part_no, description, equipment, maker, supplier_id, unit_price_usd, moq, lead_time_days, stock_port, snapshot_date.
- **must_refuse**: 'confidential supplier information' conflicts with the brief: overlaps the topic the brief asks for: 'Supplier comparisons for spares'
- **must_refuse**: 'pricing details' conflicts with the corpus: a table in the documents holds this information (column names and cell values are evidence) Evidence: column of spares_catalogue.csv: part_no, description, equipment, maker, supplier_id, unit_price_usd, moq, lead_time_days, stock_port, snapshot_date.
- **scope_out**: 'general maritime law' conflicts with the brief: refusing it would block what was asked: 'Current stock levels at ports' (The refused subject is a part of the asked task.) Evidence: ports
- **scope_out**: 'financial planning guidance' conflicts with the brief: refusing it would block what was asked: 'Order cost estimates with discounts' (The refused subject 'financial planning guidance' is a part of the asked task 'Order cost estimates with discounts'.) Evidence: Order cost estimates with discounts
- **must_refuse**: 'terms and conditions' conflicts with the brief: refusing it would block what was asked: 'Order cost estimates with discounts' (The refused subject 'terms and conditions' is a part of the asked task 'Order cost estimates with discounts'.) Evidence: Order cost estimates with discounts
- **must_refuse**: 'financial details' conflicts with the brief: refusing it would block what was asked: 'Order cost estimates with discounts' (The refused subject 'financial details' is a needed input for the asked task 'Order cost estimates with discounts'.) Evidence: Order cost estimates with discounts
- **must_refuse**: 'contractual details' conflicts with the brief: refusing it would block what was asked: 'Order cost estimates with discounts' (The refused subject 'contractual details' is a part of the asked task 'Order cost estimates with discounts'.) Evidence: Order cost estimates with discounts
