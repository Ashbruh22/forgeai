# Requirements: Brightwater Assistant

**Purpose.** Provide accurate information to Brightwater Library visitors about loans, fines, membership, book availability, and cafe menu.

**Audience.** Brightwater Library visitors

**Tone.** The assistant should use a friendly, helpful, and informative tone to provide information about library services, book availability, and cafe menu details.

## Scope

- in: loan periods
- in: fines
- in: library card membership
- in: book availability
- in: cafe menu
- out: personal account inquiries

## Must answer

- **T1** Are there any fines for overdue books? (priority high, coverage SUPPORTED), evidence: library_rules.md#c1
- **T2** What is the loan period for books? (priority high, coverage SUPPORTED), evidence: library_rules.md#c0
- **T3** Can I renew my library card? (priority high, coverage PARTIAL), evidence: library_rules.md#c0
  - supported part: Renewal eligibility for library card members
- **T4** Is the book 'Pride and Prejudice' available? (priority high, coverage SUPPORTED), evidence: books.csv#c2
  - answers come from a snapshot dated 2026-11-20
- **T6** tell visitors whether a specific book is on the shelf right now (priority high, coverage PARTIAL), evidence: books.csv#c7, books.csv#c4, books.csv#c2, books.csv#c8
  - supported part: Recorded shelf location and copy count, as of 2026-11-20

## Must refuse

- **R1** personal data: Excluded by the requester. Redirect: I do not have information on this. Please contact the relevant team directly.
- **R2** legal advice: Excluded by the requester. Redirect: I do not have information on this. Please contact the relevant team directly.

**Capabilities needed.** document_qa

## Gaps

- **G-T5** [unsupported_topic] The brief asks for 'What are the cafe menu options?', but none of the 3 probe questions could be answered from the documents. Action: `refuse_with_redirect`
- **G-LIVE-1** [live_data] The brief asks for current or real-time information, but the documents are a static snapshot and no tool provides it: 'It should also tell visitors whether a specific book is on the shelf right now' The recorded version (Recorded shelf location and copy count) IS answerable from the documents as of 2026-11-20; only the live value is refused. Action: `refuse_with_redirect`

## Assumptions

- **scope_out**: The assistant should avoid answering topics not explicitly covered by the documents, such as specific personal account inquiries, detailed room booking procedures, or non-library-related cafe policies not mentioned in the provided materials. (ASSUMPTION (non-interactive mode): Q1 What topics related to the library, cafe, or membership should the assistant avoid answering, even if they seem connected to the provided documents?)
- **must_refuse**: The assistant should refuse to answer questions involving personal data, legal advice, or any information that is not covered by the specified topics such as loan periods, fines, library card membership, book availability, or cafe menu details. (ASSUMPTION (non-interactive mode): Q2 What kinds of questions should the assistant always refuse to answer, such as personal data or legal advice?)
- **tone**: The assistant should use a friendly, helpful, and informative tone to provide information about library services, book availability, and cafe menu details. (ASSUMPTION (non-interactive mode): Q3 What tone should the assistant use when providing information about library services, book availability, and cafe menu details?)
- **gaps**: refuse_with_redirect for every gap (ASSUMPTION (non-interactive mode): Q4 The documents and tools cannot support these requests: What are the cafe menu options?; Current availability of the book on the shelf. Should the assistant refuse them with a redirect, or will you add documents? Reply 'refuse' to have the assistant refuse and redirect, or 'documents' if you will add documents.)

## Assumptions rejected (brief > corpus > assumptions)

- **scope_out**: 'cafe policies' conflicts with the brief: overlaps the topic the brief asks for: 'What are the cafe menu options?'
- **must_refuse**: 'loan periods' conflicts with the brief: overlaps the topic the brief asks for: 'What is the loan period for books?'
- **must_refuse**: 'fines' conflicts with the brief: overlaps the topic the brief asks for: 'Are there any fines for overdue books?'
- **must_refuse**: 'library card membership' conflicts with the brief: overlaps the topic the brief asks for: 'Can I renew my library card?'
- **must_refuse**: 'book availability' conflicts with the brief: overlaps the topic the brief asks for: 'Are there any fines for overdue books?'
- **must_refuse**: 'cafe menu details' conflicts with the brief: overlaps the topic the brief asks for: 'What are the cafe menu options?'
- **scope_out**: 'room booking procedures' conflicts with the corpus: the documents answer questions about it (SUPPORTED); excluding it would refuse something the corpus supports Evidence: rooms.csv#c4: hourly_fee_usd: 60 | rooms.csv#c1: room: Reading Room | capacity: 40 | hourly_fee_usd: 0 | projector: no
