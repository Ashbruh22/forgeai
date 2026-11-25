# Requirements: HRPolicyHelper

**Purpose.** Answer HR policy questions for shore staff on leave, travel claims, conduct rules, and salary slips.

**Audience.** Shore-based employees and staff

**Tone.** Professional and clear

## Scope

- in: Leave policy
- in: Travel claims
- in: Conduct rules
- in: Salary slip questions
- out: ethical dilemmas

## Must answer

- **T1** Annual leave entitlements (priority high, coverage SUPPORTED), evidence: leave_policy.pdf#c3, leave_policy.pdf#c4
- **T2** Travel expense reimbursement procedures (priority high, coverage SUPPORTED), evidence: travel_expense_policy.pdf#c10, travel_expense_policy.pdf#c8
- **T3** Code of conduct rules (priority high, coverage SUPPORTED), evidence: code_of_conduct_it_policy.pdf#c3, code_of_conduct_it_policy.pdf#c6

## Must refuse

- **R1** ethical dilemmas: Excluded by the requester. Redirect: I do not have information on this. Please contact the relevant team directly.

**Capabilities needed.** document_qa

## Gaps

- **G-T4** [unsupported_topic] The brief asks for 'Salary slip interpretation', but none of the 3 probe questions could be answered from the documents. Action: `refuse_with_redirect`

## Assumptions

- **scope_out**: The assistant should avoid answering questions related to sensitive personal data, legal advice, or matters outside the scope of the provided policies, such as salary slip details that may involve confidential information. Additionally, the assistant should not provide guidance on ethical dilemmas or misconduct reporting, which are covered under compliance policies. (ASSUMPTION (non-interactive mode): Q1 What specific topics related to HR policies or salary slips should the assistant avoid answering or assisting with?)
- **must_refuse**: The assistant should refuse to answer questions involving personal data requests, legal advice, or any matters outside the scope of HR policies such as leave, travel claims, and salary slips. It should also avoid providing guidance on ethical dilemmas or misconduct reporting unless explicitly covered in the provided policies. (ASSUMPTION (non-interactive mode): Q2 What specific types of questions should the assistant always refuse to answer, such as personal data requests or legal advice?)
- **gaps**: refuse_with_redirect for every gap (ASSUMPTION (non-interactive mode): Q3 The documents and tools cannot support these requests: Salary slip interpretation. Should the assistant refuse them with a redirect, or will you add documents? Reply 'refuse' to have the assistant refuse and redirect, or 'documents' if you will add documents.)

## Assumptions rejected (brief > corpus > assumptions)

- **scope_out**: 'sensitive personal data' conflicts with the corpus: the documents answer questions about it (SUPPORTED); excluding it would refuse something the corpus supports Evidence: code_of_conduct_it_policy.pdf#c5: All employees are required to maintain the confidentiality of all company and client information, including but not limited to operational data, financial records, and communication logs, throughout their employment and for 2 years after employment ends. | code_of_conduct_it_policy.pdf#c8: 7.1 All personal devices used for company business must be registered with IT prior to use, and must remain registered throughout their active use period.
- **scope_out**: 'legal advice' conflicts with the corpus: the documents answer questions about it (PARTIAL); excluding it would refuse something the corpus supports Evidence: code_of_conduct_it_policy.pdf#c5: Any breach of confidentiality, regardless of intent, will result in disciplinary action, up to and including termination of employment and legal prosecution under applicable laws.
- **scope_out**: 'confidential information' conflicts with the corpus: the documents answer questions about it (SUPPORTED); excluding it would refuse something the corpus supports Evidence: code_of_conduct_it_policy.pdf#c5: All employees are required to maintain the confidentiality of all company and client information, including but not limited to operational data, financial records, and communication logs, throughout their employment and for 2 years after employment ends. | code_of_conduct_it_policy.pdf#c5: Employees must not disclose any confidential information to third parties, including but not limited to former colleagues, contractors, or external auditors, without prior written authorization from the Chief Compliance Officer.
- **scope_out**: 'misconduct reporting' conflicts with the corpus: the documents answer questions about it (SUPPORTED); excluding it would refuse something the corpus supports Evidence: code_of_conduct_it_policy.pdf#c6: The Ethics Hotline will acknowledge receipt of all reports within 5 working days, regardless of the nature or severity of the concern. | code_of_conduct_it_policy.pdf#c6: All reports of misconduct must be submitted through the Ethics Hotline, which is the designated internal channel for such concerns.
- **must_refuse**: 'personal data requests' conflicts with the corpus: the documents answer questions about it (SUPPORTED); excluding it would refuse something the corpus supports Evidence: leave_policy.pdf#c5: Annual leave requests need 10 working days notice to allow adequate planning. | code_of_conduct_it_policy.pdf#c9: Employees are prohibited from using personal email accounts to conduct company business or to communicate with clients, suppliers, or colleagues unless explicitly authorized by a manager.
- **must_refuse**: 'legal advice' conflicts with the corpus: the documents answer questions about it (PARTIAL); excluding it would refuse something the corpus supports Evidence: code_of_conduct_it_policy.pdf#c5: Any breach of confidentiality, regardless of intent, will result in disciplinary action, up to and including termination of employment and legal prosecution under applicable laws.
- **must_refuse**: 'HR policies' conflicts with the corpus: the documents answer questions about it (SUPPORTED); excluding it would refuse something the corpus supports Evidence: leave_policy.pdf#c5: Annual leave requests need 10 working days notice to allow adequate planning. | code_of_conduct_it_policy.pdf#c3: All employees are expected to conduct themselves with professionalism, integrity, and respect in all interactions with colleagues, clients, and third parties at all times.
- **must_refuse**: 'misconduct reporting' conflicts with the corpus: the documents answer questions about it (SUPPORTED); excluding it would refuse something the corpus supports Evidence: code_of_conduct_it_policy.pdf#c6: The Ethics Hotline will acknowledge receipt of all reports within 5 working days, regardless of the nature or severity of the concern. | code_of_conduct_it_policy.pdf#c6: All reports of misconduct must be submitted through the Ethics Hotline, which is the designated internal channel for such concerns.
