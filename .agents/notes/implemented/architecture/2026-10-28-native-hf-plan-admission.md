# Decision: native numerical admission for the fixed progressive HF plan

Status: implemented
Date: 2026-10-28

The missing-library/protocol allowance is superseded by
[the current-protocol requirement](../compatibility/2026-10-29-required-hf-policy-protocols.md).
The independent policy and arbitrary-width integer behavior remain unchanged.

## Problem

The bounded HF controller already delegates execution scheduling, final scalar
verification and qualified CPU projection/physical audits to native owners.
Its typed plan still decided convergence ordering and configured work/cost
budgets in Python, so native SDK users could not query the same planning policy.
This is a separate R4/R6 increment under #2227 from the open force/AOT/provider
ownership PRs checked on 2026-10-28.

## Decision

`generativeqc_hf_plan_query_v1` owns numerical admission of the existing fixed
source/target plan. It returns all unmet gates, without changing any requested
solver control: source energy/density thresholds cannot be stricter than the
target, source/total iteration counts must fit, and a cost-limited plan needs
both finite positive estimates whose binary64 sum fits. Equality is admissible.
An overflowing integer sum cannot wrap into budget; an overflowing floating
cost sum fails a finite budget.

Python retains typed graph/model/evidence validation, diagnostic wording and
the previous error precedence. It consumes one native mask at the original
validation points. The calculator builder supplies its selected target library
through a non-retained dataclass `InitVar`; that argument is absent from equality,
serialization and plan identity. Standalone plans use only the first configured
existing library. No native error or refusal retries Python policy.

Missing protocols and Python integer counts outside the uint64 ABI retain
explicit compatibility. Arbitrary-width values must not truncate through ctypes.
This query neither authenticates the scientific model nor admits a complete
calculation. Strict target arithmetic, physical verification, actual work
accounting and independent observable accuracy remain mandatory separate gates.

## Rejected alternatives

- Discovering a different library when the selected calculator lacks the new
  symbol would split runtime ownership and hide compatibility behavior.
- Retaining a library in the immutable scientific plan would make model identity
  and serialization depend on process-local execution state.
- Removing compatibility would break native-free typed planning and older
  installed libraries without scientific benefit.
- Extending this change to force/AOT/provider lifecycle would overlap concurrent
  PRs and require a different qualification matrix.

## Evidence

`tests/native/test_hf_plan.cpp` exercises the public ABI, exact and below-budget
limits, overflow, ignored absent fields, malformed input, atomic output and
overlapping storage. `tests/python/test_native_hf_plan.py` compiles the actual
adapter and checks independent arbitrary-width arithmetic, concurrent detached
decisions, plan identity, explicit compatibility and rejection without retry.

The existing progressive controller endpoint suite forbids the compatibility
planner while comparing CPU RHF/UHF Direct/DF energy and forces with PySCF.
The [retained endpoint receipt](../../../../benchmarks/results/hf-plan-admission-20261028/README.md)
records 672 CPU samples with identical scientific/work signatures and all eight
median ratios at or below 1.004344 (gate 1.05), plus 126 passing CTests and 261
focused Python tests with 25 subtests. It does not qualify CUDA science, a general
adaptive controller, or completion of #2227.

## Revisit when

A native prepared controller owns general stage graphs or per-item adaptive
scheduling. Reuse this policy and preserve explicit model/budget contracts;
do not treat numerical admission as evidence of final target accuracy.
