# Decision: admit the dense native MP2 orbital oracle before materialization

Status: implemented
Date: 2026-10-10

## Problem

The sixth retained materialization in the native review inventory is
`initial_orbital_weights::result.two_electron`. Checked extent arithmetic
prevented wraparound but did not establish a size or byte admission policy.
The dense RHS route allocated a Fock matrix and a complete weight tensor,
then retained them through multiple rotation-gradient contractions. The
relaxed-weight caller also kept both RHS vectors while allocating overlap
and stationarity scratch. Bounding the N^4 vector alone misses these peaks.

## Decision

Keep this dense legacy/oracle representation legal only for at most 12 orbitals,
matching the existing small complete-gradient integral oracle's 12-AO domain.
Use a default 1 MiB newly owned numeric budget and checked, success-path allocation-free
planners. Reject oversized dimensions before input walks, extent products or
numeric allocation; reject insufficient bytes before entering the old numeric
body. Admitted values, contraction order, result layout and existing C++ entry
point symbols remain unchanged. Explicit `*_with_budget` entries can reduce
the allowance, but cannot raise the hard size cap.

The original RHS caller/producer remains visible to the conservative source
scanner. No candidate is removed or relabeled as required/retired. Rebind the
review manifest to the actual changed source and retain the remaining endpoint,
consumer and profitability questions. This is the size-gate requirement of
#1626, not closure of its six retention decisions.

## Owned storage invariants

For N orbitals and O*V occupied-virtual pairs, the RHS returns
`8*(N^4+N^2+2*O*V)` numeric bytes. Its peak is
`8*(N^4+4*N^2+O*V)`: the second rotation call owns a new gradient before
move assignment retires the old gradient. The relaxed-weight peak instead is
`8*(N^4+4*N^2+2*O*V)`, with both RHS vectors still live alongside the final
gradient, overlap and stationarity. Its returned numeric bytes are
`8*(N^4+2*N^2)`. The independent scoped allocation probe verifies these peaks.

Caller-owned h/ERI/adjoint/Z input vectors are excluded explicitly, as are
allocator rounding, object headers and exception-message storage. Do not use
this owner-only reservation as a complete endpoint/RSS claim. Failed guards
may allocate exception storage; they are not zero-heap-allocation receipts.

The production `canonical_orbital_rhs_streamed` and relaxed streamed ownership
remain independent, do not allocate the dense N^4 weight vector, and do not
inherit this small-oracle cap. No hidden fallback or Python/PySCF work is added.

## Rejected alternatives

A byte bound alone would still allow expensive dense rotation work at larger
dimensions. A guard inside `initial_orbital_weights` would be too late to cover
the Fock allocation and cannot admit the complete relaxed caller. Rewriting
the dense result as a factorized view without qualifying retaining consumers
would change the oracle ABI and conceal the materialization. A renamed bypass
or a whole-file oracle exemption would defeat the conservative inventory.

## Evidence and revisit conditions

The native probe compiles the complete actual translation units and only drops
unreferenced link sections, without replacing numeric functions with doubles.
Independent Python/NumPy tensor algebra validates all returned fields using
synthetic canonical Hamiltonians and committed PySCF fixtures. Exact-bound
execution is compared for exact returned-value equality with the legacy entry
point. Resource tests
measure requested-payload lifetimes, one-byte-short rejection, oversized and
integer-limit dimensions, and empty invalid layouts. Validation results and
source-matched materialization review are recorded after qualification.

Revisit the hard cap only with an explicitly bounded larger oracle workload,
independent parity and measured work/storage evidence. Retiring the dense
representation or any of the five shell stages additionally requires real
consumer/physical derivative parity and complete endpoint profitability
comparisons. No performance, whole-method or issue-closure claim follows here.

## Source-matched qualification

The final source freeze is tree `7003f2ccb48f70e6bfdc34177132cdc606914dcb`,
based on detached commit `f87d51ab616cc9aa774bd344a15e268cef7eaf05`, with
9,204 entries. Local verification and remote verification before and after the
full build report no mismatches. The archive SHA-256 is
`6fe26ca5012ed31ff18b3c2d4a2e7dba3a0066fc1614073ce222ce6a37cf77a1`.
This appendix and the equality-wording clarification above postdate the freeze;
they do not change the qualified native implementation or tests.

Retained evidence is under
`.artifacts/issue1626-dense-oracle/final-format/` in the detached workspace:

- `qualified-tests.log`: **209 passed, 48 subtests passed**, covering dense
  admission, derivative pullback, streamed workspace, representation census,
  diagnostics, materialization review and the structured scanner. Temporary
  fixtures are outside Git. The earlier in-checkout fixture attempt caused two
  provenance tests to observe a dirty Git root instead of a non-Git fixture;
  its failed log is preserved separately, without weakening test expectations.
- `probe-summary.json` and `probes/*.json`: 16 independently checked RHS/relaxed
  observations on four synthetic and four committed Hamiltonians, with every
  returned field inside `atol=3e-11`, `rtol=3e-11`. Maximum absolute error is
  `4.973799150320701e-14`. All exact-budget returned values equal the legacy
  entry point. At N=12, O=6, requested-payload peaks are 170,784 bytes (RHS)
  and 171,072 bytes (relaxed), exactly the planned peaks. Output lifetimes end
  at zero tracked live bytes and no request journal entries are dropped.
  Probe SHA-256:
  `91691c8999c5df837c37eef8f30ccadbeef8f9ecc1cb12abf8536c8cd6df2c84`.
- `rejections.json`: 32 raw zero/short-budget, oversized/integer-limit,
  invalid-dimension and malformed/nonfinite-response receipts. Rejected guards
  may allocate exception storage; these are not zero-heap-allocation claims.
- `native-qualified/`: Release sm_120, CUDA 12.9, AOT-disabled full native build,
  CMake cache, Ninja commands, logs, contracts executable and library. Existing
  ccache 4.5.1 is reused. Slurm job **6959**, on `node1`, partition `main`,
  `gpu:5090:1`, with a finite ten-minute limit and assigned device visibility
  preserved, completes with `ExitCode=0:0` and
  `MP2 native gradient contracts passed`. These existing finite-difference,
  dense/streamed, DF and resource contracts do not establish GPU physical-force
  parity or complete endpoint profitability. Completed controller state is
  retained because cluster accounting is disabled.
- Qualified library SHA-256:
  `9767c63e12770f6704a498525ab024b6f1115ee48a9e10dd7af48d01884de1c2`;
  contracts executable SHA-256:
  `8e66853866d217ea95c217c88a104366c1f847b53647261207574be6f2897406`.
- `materialization-review.json`: **PASS**, baseline 6 / candidate 6, added 0,
  removed 0, changed 6, with no review errors. All six dispositions remain
  `retained-pending-evidence`. Ruff, focused clang-format and `git diff --check`
  also pass.

The retained read-only snapshot still reports #1626 **OPEN**. Source and binary
hashes establish artifact integrity, not scientific dependency lineage. This
completes only the explicit native dense-oracle admission gap; it neither
retires a materialization nor closes the remaining consumer, physical-derivative,
joint endpoint-admission or profitability requirements.
