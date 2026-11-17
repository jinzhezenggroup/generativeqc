# Decision: audit scoped dispatch edges separately from native scientific kernels

Status: implemented
Date: 2026-11-14

## Problem

The previous #2227 delivery snapshot identified source
`9cfb77727661cc64e47938d9d0ba65a157427439`. Subsequent public CPU full-range
PBE0/B3LYP Hessian work and D3(BJ) curvature broaden scientific capability,
but their molecular controllers remain Python-owned. Reading native source
presence or numerical acceptance as a complete runtime cutover would erase
real R3/R6 work. Conversely, the dedicated CUDA WB97M-V second-order frontend
does bypass those controllers and refuses missing AOT closure before SCF.

The historical matrix alone also misses newer review domains: the CPU
second-order row originally covered Cartesian closed-shell LDA/PBE, not the
current spherical/full-range/UKS and correction consumers. Rewriting that
baseline would destroy its historical meaning. A flat list of source anchors
cannot distinguish an admitted native branch from a retained Python branch.

## Decision

Extend the existing separate delivery snapshot to schema v2 rather than
create another registry. Keep all 20 baseline IDs and the original baseline
artifact unchanged. Bind the refreshed source review to
`150b81348914bd8fd186050ffdc1c3c35c4d4b0c` and record six explicit current
review axes per row. Axes are not Cartesian products of capabilities.

Record contextual caller/callee file edges alongside retired work, remaining
controllers and unmet acceptance. Classify each Python edge independently:
wrapper, compiler/AD, scientific runtime, compatibility or reference oracle.
A null role denotes a wholly native edge. The installed CLI therefore has
no invented Python wrapper, while the dedicated CUDA second-order execution
row no longer suggests that Python compiler AD owns its numerical dispatch.

The checker enforces complete baseline membership, local source evidence,
current-axis shape, edge/row classification consistency and honest remaining
work. Symlinks may navigate within the checkout, not substitute external
source for a local evidence anchor. Validation deliberately does not inspect
production bodies, require particular symbols/AST shapes, match HEAD, or
infer scientific support from graph edges. Behavioral tests remain the means
of checking actual dispatch, refusal and lifetime semantics.

## Important source findings

- Per-item native prepared force admission is not a method-name whitelist.
  Non-admitted named compatibility paths differ from a selected native error;
  automatic CUDA force refusal cannot enter Python stationary/JIT science.
- Public CPU full-range and WB97M-V second order still use Python source,
  response and scalar-column orchestration. The optional uncut two-body
  D3(BJ), ATM-off correction is composed once in Python and builds compiler
  AD/TensorIR at runtime; an artifact miss can compile. Native curvature
  arithmetic is not a complete installed Python-free AOT endpoint.
- Native precision decisions/stage ordering do not retire the Python facade's
  candidate-fleet, auditor and seed orchestration. Shared Standard/Fast
  construction and target-based optimization do not change that ownership.
- Native CBS weights do not retire Python component grouping, budget
  distribution or observable composition. Native resource composition does
  not make all domain inventories or provider/RSS lifetimes native-owned.

## Evidence and boundaries

Run the ownership checker and the ownership/retirement tests. Existing
host-only native-Hessian frontend, prepared-resource and prepared-force-query
tests exercise actual public adapters with injected native calls; CLI launcher
and initial-guess ABI tests cover refusal and native transport boundaries.
These tests are not molecular numerical or real-GPU execution evidence.

No production equation, numerical threshold, capability admission, compiler
artifact identity, baseline source SHA or independent reference is changed.
No new GPU qualification, performance result, R1–R6 completion or parent issue
closure follows from this R0 audit. Listed ledger tests remain evidence
locations, not claims that every listed qualification was rerun.

## References

- [Original separate-snapshot decision](2026-11-04-native-runtime-delivery-audit.md)
- `manifests/maintenance/python_native_runtime_delivery.json`
- `tools/check_native_runtime_ownership.py`
- `tests/python/test_native_runtime_ownership.py`
