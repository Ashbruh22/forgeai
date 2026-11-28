---
version: 1
---
# system
You write realistic questions that users might ask an assistant. The questions are used to test
that the assistant does NOT improvise an answer when it lacks a capability.
# user
Assistant purpose: {{purpose}}
A capability this assistant does NOT have: {{tool}}

Write 2 different realistic questions a user of this assistant might ask that would need that
capability to answer. Return them as `questions`.
