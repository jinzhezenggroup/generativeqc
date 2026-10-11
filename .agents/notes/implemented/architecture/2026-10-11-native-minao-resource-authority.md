# Decision: MINAO numeric resource capacity belongs to the native runtime

Status: implemented (pending exact-head qualification)
Date: 2026-10-11

## Problem

The Python MINAO resource planner and the native preliminary-SCF preparation
duplicated the exact same numeric-payload capacity polynomial and element-specific
primitive counts. Changes to the native validator, occupied-ANO table or numerical
workspace could silently leave the public resource budget under-reserved.

## Decision

The production native formula lives only in
`src/scf/preliminary_guess.cpp::minao_numeric_capacity_shape`. The ordinary
`preliminary_numeric_capacity` gate and the additive, shape-only private C
function `generativeqc_resource_minao_numeric_capacity_v1` use it, with
source-AO and primitive counts read from the native occupied-ANO tables.

Python's `with_initial_guess_resources` asks the live library for the numeric
capacity whenever the new symbol is present. It continues to account for all
retained seeds and only the largest serialized preparation workspace alongside
the target owner. The preceding Python polynomial is retained **only** as a
compatibility path for older binary libraries and pure data-contract tests;
the production path with a current native binary does not execute it.

No source basis, integral evaluation, Fock matrix, CUDA device, native context,
or target SCF is constructed by the query. Small source-element metadata is
validated on the CPU. A failed query returns a nonzero status and leaves the
caller-supplied output untouched, rather than downgrading to an unverified bound.

## Rejected alternatives

- Copy new formula coefficients into Python: preserves the drift hazard.
- Prepare a dummy target system/SCF merely to learn memory: costs unnecessary
  allocations, can trigger hardware selection, and confuses estimate with execution.
- Broaden the stable public C ABI during a private planner refactor: unnecessary;
  this query can be versioned additively within the resource owner.
- Remove old-library support immediately: risks breaking users who deliberately
  retain an older native binary with the existing MINAO schema.

## Invariants

- Source AO/primitive counts are owned by the native occupied-ANO records.
- The budget remains a conservative numeric-payload bound, **not** process RSS,
  full-target memory or allocator overhead.
- Zero, unsupported H-Ar, null/invalid layout and overflow fail closed.
- The target's basis, functional, precision, SCF policy, fallback behavior and
  candidate seed are unchanged. No new GPU or force qualification is implied.
- Python references are compatibility only, never a second scientific runtime.
  Future removal requires an intentional old-binary support policy.

## Evidence required before merge

1. Compile native CPU/CUDA libraries and the native preliminary-guess CTest.
2. Check native real-system bound against the shape query and old reference
   across H-Ar, small/large AO counts and composition variants.
3. Run Python resource-planner suites and the boundary/failure tests. Verify
   the current binary actually takes the native query rather than the fallback.
4. Confirm invalid topology leaves output untouched and the same seed limit
   accepts exact capacity but declines capacity minus one.
5. Confirm tests in CPU-only environments do not initialize CUDA.

No performance speedup or numerical regression result is claimed until these
tests complete.

## References

- #926 (shared execution paths), #933 (resource/lifecycle contract), #934
  (cutover evidence), and this implementation branch.
