# Audit: shared ownership is not Standard/Fast execution parity

Status: proposed; remaining native optimizations are not promoted
Date: 2026-11-05
Audited base: `0a214362fa9d762cf0a3a6be821d7cb5d185f257`
Ownership change: `08d9eccd902e39325ea94a0d94ee74e57f5f5157`

## Scope

The ownership change shares scalar/fleet drivers, benchmark entry points and
successful unaudited candidate lifetime. It does not establish that Standard
and Fast differ only by FP32 versus FP64. This follow-up inspects candidate
construction, native scheduling, initial guesses, final-Fock reuse, incremental
operators, tiling, resource admission and independent audit transitions.

## Required numerical differences

- Fast may select provisional Fock/XC/gradient arithmetic, stopping limits and
  source screening from its application budget. Strict physical publication
  remains required; a strict returned state does not imply a strict trajectory.
- Fast qualification and insufficient evidence can require an independent FP64
  audit and bounded replacement verification. Standard without an explicit
  target runs once. Removing this difference would weaken scientific evidence.
- Mixed operators cannot borrow the existing exact incremental anchor without
  a composed numerical qualification. The native incremental admission uses
  actual precision capabilities, not the Python profile spelling.
- Initial-guess ownership and RHF converged-Fock reuse are shared. Existing
  strict physical-state and seed-provenance checks still apply to both.

## Confirmed non-dtype execution differences

1. `PrecisionCalculator.__init__` constructs a strict model/capability probe
   before constructing the selected CUDA Fast candidate. Standard normally
   keeps that first calculator. This is control-plane setup, not evidence of
   extra electronic solves, but cold setup is not structurally identical.
2. `admit_fast_anchor_delta` in `src/dft/application_precision.hpp` admits fixed
   **FP64** J/K anchors only for `budget_fock && budget_stationarity` WB97M-V
   outer solves. Standard has no equivalent default. The separate strict-anchor
   PR #2986 was open/draft at this audit and must not be treated as merged or
   default-qualified. Do not copy its research timings onto this PR.
3. The large fitted/nonlocal automatic force-tile branch in
   `src/methods/dft_method.cpp` requires an all-zero application budget before
   enlarging the native force tile. Its other predicates include strict SCF,
   model/binding, geometry, device and memory conditions. A nonzero budget alone
   is not proof that this force consumer executes mixed arithmetic.
4. The same file's optional numeric-capacity reservation rejects any nonzero
   application budget. Its proof currently describes FP64 storage without
   mixed scratch. Preserve the bounded fallback; sharing this admission first
   requires an actual selected-operator capacity proof, not deletion of the gate.

These are source-level facts, not a measured performance difference on every
endpoint. Tiling changes can change work/reduction order; fixed-anchor changes
have separate strict-tail and screening obligations. No native default or
resource promise is changed by this follow-up without real-device qualification.

## Corrected benchmark assumption

`benchmarks.ks_incremental_default` and its CUDA qualification previously
expected automatic incremental J/K only when the arm was named Standard. That
incorrectly rejects Fast with a strict candidate. The expectation now follows
the admitted budget-Fock route, direct versus DF provider and seed phase within
the campaign's existing default-grid hybrid/size domain. It does not infer
eligibility from zero mixed work in a warm call, nor relax the off/DF/mixed
exclusions. The added strict-budget Fast CUDA case still requires its ordinary
independent observable audit and complete endpoint oracle.

## Validation and retained provenance

The focused policy/native/anchor/tile/incremental suite passed 522 tests, with
15 CUDA cases skipped. The retained native Libxc/capability catalog suites passed
70 tests. Ruff checks/formatting and Sphinx HTML with warnings as errors passed.
The new CUDA case is coverage to execute on a usable scheduled device, not
completed real-device qualification.

The ownership change also changes source identities because the inventories
include runtime Python. The official native Libxc catalog generator refreshed
only the current checked source identities in the retained snapshot and rendered
report. The historical campaign, native binary identity, capability summary and
already-stale status remain unchanged; this is not numerical requalification.

## Acceptance before claiming precision-only differences

- Resolve candidate configuration once without losing selected-model checks.
- Bind optional reuse, tile and resource choices to actual operator precision,
  shape and bounded storage rather than the profile or budget's mere presence.
- Qualify shared strict and mixed alternatives independently. Preserve explicit
  off controls, minimum-storage fallbacks, warm/moved seed isolation, physical
  final-state proof, correction completeness and independent reference gates.
- Compare complete cold/warm/moved endpoints at matched controls and report
  constructor/preparation, actual Fock/force work, audit solves and memory.
- Until those gates pass, describe the current PR as shared ownership, not full
  Standard/Fast execution parity. Local Slurm had no usable GPU at this audit.
