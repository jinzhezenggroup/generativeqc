# Decision: native bounded application-precision execution order

Status: implemented
Date: 2026-11-04

## Problem

Required native admission/control queries removed duplicate Python policy
arithmetic but left candidate/retry/audit/verification order in Python. An
optional controller with a Python fallback would preserve that competing owner
and conceal mismatched installations.

## Decision

`src/runtime/precision_execution.hpp` owns one synchronous per-item schedule.
The selected library's versioned C ABI requests candidate execution, at most one
explicitly eligible numerical FP64 retry, classification, candidate-cache release,
independent audit and, only after an unmet audit, one tighter verification.
Classification/audit decisions are supplied by their existing scientific owners;
the controller does not recreate admission or observable formulas.

Callbacks have separate success/retry/error codes. Only the initial eligible
candidate can request retry. C++ exceptions become failed progress receipts;
Python preserves original exceptions and tracebacks without unwinding through
libffi. Cancellation, allocation, input and audit errors are not retry requests.
Input descriptors are copied before callbacks; invalid input leaves output
unchanged. Once work starts, attempted/completed masks expose failed progress.

Python requires the selected controller before endpoint/fleet work, refuses
duplicate operations before executing them, and checks returned progress and
disposition against actual operations before publishing. Completed scheduling
with REJECTED disposition still raises the incumbent `PrecisionBudgetError`.
Endpoint timing includes receipt validation, not just callback time.

## Boundaries and rejected alternatives

- A portable Python schedule would defeat the ownership cutover and is absent.
- Receipt success cannot imply numerical acceptance: observable measurement and
  tighter-verification failure remain distinct from scheduling completion.
- Keep qualification/gradient interpretation, observable reductions, initial
  prepared fleet execution and independent seed eligibility/history at their
  existing Python owners. Moving them simultaneously would mix a control-flow
  change with scientific/reduction or transactional-lifetime changes.
- Callbacks retain existing candidate/prepared leases and transient auditor
  cleanup, including failure paths. The allocation-free host controller does not
  claim physical resource authority, an autonomous native precision endpoint,
  installed SDK/CLI parity or new numerical domains.

## Validation and revisit conditions

Public C/C++ ABI tests cover bounded routes, malformed/overlapping descriptors,
failed progress and callback exceptions. Python tests exercise missing/refused
and inconsistent selected controllers, duplicate-work refusal, original error
propagation, concurrency and prepared candidate/auditor seed lifetimes. Physical
HF/KS tests retain pinned independent PySCF energy/full-force and finite-difference
gates. Complete-endpoint matched timing/work remains required; callback timing
alone is not performance evidence.

Revisit when existing shared native prepared owners can absorb scientific
qualification, observable reductions and seed transactions with independent
numerical, lifetime, work/resource and endpoint performance gates. Keep the full
#2227 R1–R6 scope incomplete until those cross-cutting contracts pass.

## References

- `docs/user/application_precision.md`
- `include/generativeqc/precision_admission.h`
- `manifests/maintenance/python_native_runtime_ownership.json`
- #2227 / #2972

## Subsequent numerical-owner cutover

The retained Python observable-reduction boundary described here is superseded by
[one native observable comparison owner](../numerics/2026-11-04-native-observable-comparison.md).
That follow-up independently qualifies compensated scaled RMS and moves runtime
norms/allowances, not the remaining qualification, fleet or seed controllers.
