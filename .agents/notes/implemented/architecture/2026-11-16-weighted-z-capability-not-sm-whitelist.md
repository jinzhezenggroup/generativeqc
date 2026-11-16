# Decision: weighted-Z legality follows capability and resources, not SM identity

Status: implemented locally; stock packaging extension is not qualified for merge
Date: 2026-11-16

## Problem

The same scalar-FP64 DPPP/PPPP weighted-Z implementation was restricted to SM90
twice: production companion enumeration returned no other targets, and CUDA IR
validation rejected them. This encoded the original H100 qualification scope as
an algorithm limitation. A stock portable manifest could declare the capability
on another target but never generate it.

## Decision

Remove both architecture-name checks. Retain manifest capability admission,
validated target schedules, the fixed 96/192-thread restricted value mapping,
the DPPP/PPPP mathematical boundary and all existing exact coefficient-bit,
precision, resource and workload guards. Native weighted-Z selection remains
explicit opt-in; automatic shared-XY selection remains independent.

Make the scientific fixture's generation target explicit and record its complete
target metadata. A test generated with SM90 metadata and compiled for another
GPU must not be silently presented as that GPU's target-plan qualification.

## Rejected alternatives

- Expanding the architecture whitelist only reproduces the same maintenance
  problem for the next device.
- Enabling every optional Fock variant or copying another card's preferences
  conflates reusable algorithms with packaging cost and measured profitability.
- Adding portable companions to every tuned profile without an explicit
  capability decision changes module size and first-use costs unnecessarily.

## Invariants

No change to emitted recurrence arithmetic, screening, canonical scatter,
coefficient equality, bounded task fallback, runtime selectors or numerical
acceptance thresholds. No new default route or cross-device performance claim.
An unavailable compiled target or CUDA image is not repaired by this change.

## Evidence

Cross-target executable tests cover catalog and conservative targets, manifest
capability refusal, fixed plan mappings, resource refusal, scientific-family
refusal, selector identity and existing native dispatch/fallback contracts.
The 33-module regression set passed 873 tests with one existing opt-in skip. One
native-library source/ABI probe was explicitly deselected because this checkout
has no built `libgenerativeqc`; its original missing-library failure is retained
in the first regression log. Ruff, whitespace and compiler-ownership checks pass.

CUDA 12.9.1 compiled the independently referenced fixture for SM80/86/87/89/90/
100/103/120/121. These are compilation/resource observations, not execution on
those nine physical architectures. SM110 remains target-plan-only coverage with
this installed compiler.

Slurm job 17578 on n1 executed the SM120 fixture on RTX 5090. Job 5322 on n2
replayed the identical authenticated binary on RTX PRO 6000 Blackwell. Each
passed 96 matrix predicates, 56 exact guard controls and memcheck/racecheck/
synccheck with zero reported errors/hazards. Absolute/relative Fock tolerances
remain 2e-11/2e-10. The earlier n2 compile job 5321 failed because its host compiler
could not find `cc1plus`; that failure is retained, not relabelled as a pass.

On both tested SM120 cards, XY and weighted-Z each use 166 registers; weighted
DPPP/PPPP retain 96/72B local storage and use 1664/1152B static shared storage.
This is fixture evidence, not a complete HF energy/force or cold/warm/moved
performance qualification, and not A100/H100 execution evidence.

The existing stock default checker correctly refuses the expanded compiled
masks (`sm_100: fresh global-default promotion campaign required`, with other
portable targets also changed). Runtime-default-off does not eliminate module
size/first-use cost. Its validator, retained decisions and frozen effective
policy bytes remain unchanged. Do not merge or describe the stronger stock
packaging as accepted until the required complete-endpoint campaign passes.

Raw commands, independent arrays, target receipts, compiler/cache logs, Slurm
visibility, binary hashes, sanitizer output and the failed first n2 compile are
retained locally in `/data/jzzeng/weighted-z-portability-validation-20261116/`.
They are not public release artifacts or global-default qualification receipts.

## References

- Supersedes the architecture-bound admission of the
  [original compiler decision](../performance/2026-11-07-exact-weighted-rys-z-companion.md)
  and [endpoint opt-in decision](../performance/2026-11-07-weighted-z-endpoint-opt-in.md),
  not their historical measurements.
- The raw-K experiment in `2026-11-08-dft-streaming-shared-rys-xy.md` remains
  distinct from paged-HF endpoint qualification.
