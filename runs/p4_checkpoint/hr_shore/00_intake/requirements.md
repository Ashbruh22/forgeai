# Requirements: HRPolicyHelper

**Purpose.** Answer HR policy questions for shore staff on leave, travel claims, conduct rules, and salary slips.

**Audience.** Shore-based employees and staff

**Tone.** Professional and clear

## Scope

- in: Leave policy
- in: Travel claims
- in: Conduct rules
- in: Salary slip questions
- out: (nothing excluded)

## Must answer

- **T1** Annual leave entitlements (priority high, coverage SUPPORTED), evidence: leave_policy.pdf#c3, leave_policy.pdf#c4
- **T2** Travel expense reimbursement procedures (priority high, coverage SUPPORTED), evidence: travel_expense_policy.pdf#c10, travel_expense_policy.pdf#c8
- **T3** Code of conduct rules (priority high, coverage SUPPORTED), evidence: code_of_conduct_it_policy.pdf#c3, code_of_conduct_it_policy.pdf#c6

## Must refuse

- (none)

**Capabilities needed.** document_qa

## Gaps

- **G-T4** [unsupported_topic] The brief asks for 'Salary slip interpretation', but none of the 3 probe questions could be answered from the documents. Action: `refuse_with_redirect`

## Assumptions

- **scope_out**: The assistant should avoid answering questions related to sensitive personal data, legal advice, or matters outside the scope of the provided policies, such as salary slip details that may involve confidential information. Additionally, the assistant should not provide guidance on ethical dilemmas or misconduct reporting, which are covered under compliance policies. (ASSUMPTION (non-interactive mode): Q1 What specific topics related to HR policies or salary slips should the assistant avoid answering or assisting with?)
- **must_refuse**: The assistant should refuse to answer questions involving personal data requests, legal advice, or any matters outside the scope of HR policies such as leave, travel claims, and salary slips. It should also avoid providing guidance on ethical dilemmas or misconduct reporting unless explicitly covered in the provided policies. (ASSUMPTION (non-interactive mode): Q2 What specific types of questions should the assistant always refuse to answer, such as personal data requests or legal advice?)
- **gaps**: refuse_with_redirect for every gap (ASSUMPTION (non-interactive mode): Q3 The documents and tools cannot support these requests: Salary slip interpretation. Should the assistant refuse them with a redirect, or will you add documents? Reply 'refuse' to have the assistant refuse and redirect, or 'documents' if you will add documents.)

## Assumptions rejected (brief > corpus > assumptions)

- **scope_out**: 'sensitive personal data' conflicts with the brief: refusing it would block what was asked: 'Travel claims' (The refused subject is a needed input for the asked task.) Evidence: Travel claims
- **scope_out**: 'confidential information' conflicts with the brief: refusing it would block what was asked: 'Conduct rules' (The refused subject is a part of the asked task.) Evidence: Conduct rules
- **scope_out**: 'ethical dilemmas' conflicts with the brief: refusing it would block what was asked: 'Also help people with salary slip questions.' (The refused subject 'ethical dilemmas' is a part of the asked task 'help people with salary slip questions'.) Evidence: help people with salary slip questions
- **scope_out**: 'misconduct reporting' conflicts with the brief: refusing it would block what was asked: 'Conduct rules' (The refused subject is a part of the asked task.) Evidence: Conduct rules
- **must_refuse**: 'personal data requests' conflicts with the brief: refusing it would block what was asked: 'Travel claims' (The refused subject is a part of the asked task.) Evidence: Travel claims
- **must_refuse**: 'HR policies' conflicts with the brief: refusing it would block what was asked: 'Leave policy' (the refused subject is a part of the asked task) Evidence: Leave policy
- **must_refuse**: 'ethical dilemmas' conflicts with the brief: refusing it would block what was asked: 'Also help people with salary slip questions.' (The refused subject 'ethical dilemmas' is a part of the asked task 'help people with salary slip questions'.) Evidence: help people with salary slip questions
- **must_refuse**: 'misconduct reporting' conflicts with the brief: refusing it would block what was asked: 'Conduct rules' (The refused subject is a part of the asked task.) Evidence: Conduct rules
- **scope_out**: 'legal advice' conflicts with the corpus: the documents answer questions about it (PARTIAL); excluding it would refuse something the corpus supports Evidence: code_of_conduct_it_policy.pdf#c5: Any breach of confidentiality, regardless of intent, will result in disciplinary action, up to and including termination of employment and legal prosecution under applicable laws.
- **must_refuse**: 'legal advice' conflicts with the corpus: the documents answer questions about it (PARTIAL); excluding it would refuse something the corpus supports Evidence: code_of_conduct_it_policy.pdf#c5: Any breach of confidentiality, regardless of intent, will result in disciplinary action, up to and including termination of employment and legal prosecution under applicable laws.
