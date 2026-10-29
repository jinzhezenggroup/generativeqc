# Decision: require current native progressive-HF policy protocols

Status: implemented
Date: 2026-10-29

## Problem and decision

Progressive HF already queries native numerical plan admission and final
acceptance. Missing protocols nevertheless selected Python copies of those
policies. Before the first release, this old-library allowance keeps redundant
production owners and conflicts with the direct interface migration policy.

Require the current planning and final-verification protocols for bounded
integer metadata, both when a running controller supplies its library and when
a standalone call discovers the first configured image. Missing support, failed
loads, ABI mismatch, native refusal and malformed receipts remain errors. Do not
search for another image or retry the Python policy after selecting an owner.

Arbitrary-width Python budgets keep explicit arithmetic/verification, avoiding
uint64 truncation. This does not change graph/model validation, physical audit
execution, strict-cleanup requirements, final acceptance precedence or scientific
operators. It does not finish #2227 R4/R6 or add a CUDA qualification.

## Superseded choices and validation

This supersedes only old-library compatibility in the
[planning decision](../architecture/2026-10-28-native-hf-plan-admission.md) and
[final verification decision](../architecture/2026-10-26-native-hf-final-verification.md).
Their independent numerical policies and unbounded-integer behavior remain.

Public failure-injection tests require explicit errors with Python retry and
second-library discovery forbidden. Retain independent integer/cost arithmetic,
exact limits, every final observable gate, native refusal and receipt-validation
coverage. Source-bound complete RHF/UHF Direct/DF CPU endpoints cover planning,
source SCF, projection, target energy/forces, physical audit and owner cleanup.
