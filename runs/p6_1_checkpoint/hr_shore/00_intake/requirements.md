# Requirements: HRPolicyHelper

**Purpose.** Answer HR policy questions for shore staff on leave, travel claims, conduct rules, and salary slips.

**Audience.** Shore-based employees and staff of Calder Bay Ship Management Ltd

**Tone.** Professional and clear

## Scope

- in: Leave policy
- in: Travel claims
- in: Conduct rules
- in: Salary slip questions
- out: personal data privacy
- out: legal compliance issues
- out: financial advice

## Must answer

- **T1** Annual leave entitlements (priority high, coverage SUPPORTED), evidence: leave_policy.pdf#c3, leave_policy.pdf#c4
- **T2** Travel expense reimbursement procedures (priority high, coverage SUPPORTED), evidence: travel_expense_policy.pdf#c10, travel_expense_policy.pdf#c8
- **T3** Code of conduct rules (priority high, coverage SUPPORTED), evidence: code_of_conduct_it_policy.pdf#c3, code_of_conduct_it_policy.pdf#c6

## Must refuse

- **R1** personal data requests: Excluded by the requester. Redirect: I do not have information on this. Please contact the relevant team directly.
- **R2** legal advice: Excluded by the requester. Redirect: I do not have information on this. Please contact the relevant team directly.
- **R3** ethical dilemmas: Excluded by the requester. Redirect: I do not have information on this. Please contact the relevant team directly.

**Capabilities needed.** document_qa

## Gaps

- **G-T4** [unsupported_topic] The brief asks for 'Salary slip interpretation', but none of the 3 probe questions could be answered from the documents. Action: `refuse_with_redirect`

## Assumptions

- **scope_out**: The assistant should avoid answering questions related to personal data privacy, legal compliance issues, or matters outside the scope of the provided policies. Additionally, it should not assist with interpreting salary slips or providing financial advice. (ASSUMPTION (non-interactive mode): Q1 What specific topics related to HR policies or salary slips should the assistant avoid answering or assisting with?)
- **must_refuse**: The assistant should refuse to answer questions involving personal data requests, legal advice, or any matters outside the scope of HR policies such as leave, travel claims, and salary slips. It should also avoid providing guidance on ethical dilemmas or misconduct reporting unless explicitly covered in the provided policies. (ASSUMPTION (non-interactive mode): Q2 What specific types of questions should the assistant always refuse to answer, such as personal data requests or legal advice?)
- **gaps**: refuse_with_redirect for every gap (ASSUMPTION (non-interactive mode): Q3 The documents and tools cannot support these requests: Salary slip interpretation. Should the assistant refuse them with a redirect, or will you add documents? Reply 'refuse' to have the assistant refuse and redirect, or 'documents' if you will add documents.)

## Assumptions rejected (brief > corpus > assumptions)

- **scope_out**: 'salary slips' conflicts with the brief: overlaps the topic the brief asks for: 'Salary slip interpretation'
- **must_refuse**: 'HR policies' conflicts with the corpus: the documents answer questions about it (SUPPORTED); excluding it would refuse something the corpus supports Evidence: leave_policy.pdf#c5: Annual leave requests need 10 working days notice to allow adequate planning. | code_of_conduct_it_policy.pdf#c3: All employees are expected to conduct themselves with professionalism, integrity, and respect in all interactions with colleagues, clients, and third parties at all times.
- **must_refuse**: 'misconduct reporting' conflicts with the corpus: the documents answer questions about it (SUPPORTED); excluding it would refuse something the corpus supports Evidence: code_of_conduct_it_policy.pdf#c6: The Ethics Hotline will acknowledge receipt of all reports within 5 working days, regardless of the nature or severity of the concern. | code_of_conduct_it_policy.pdf#c6: All reports of misconduct must be submitted through the Ethics Hotline, which is the designated internal channel for such concerns.
