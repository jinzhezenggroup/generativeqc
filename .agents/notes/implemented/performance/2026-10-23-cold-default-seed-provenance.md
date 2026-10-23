# Decision: admit native cold guesses by provenance

Status: implemented
Date: 2026-10-23

## Problem

The guarded automatic Direct incremental proposal controller in #2696 used
`input == nullptr` to distinguish cold solves from imported or moved seeds.
The public default initial-guess owner prepares a Hückel density and passes it
as a non-null buffer, already tagged `CudaKsSeedOrigin::NativeColdGuess`.
Consequently the original automatic request could allocate incremental storage
and report `requested=true` while bypassing every delta on the normal cold path.
Explicit opt-in did not have this defect, so #2691's timings could not detect it.

The new preparation logic also broke an older host controller fixture that
copied a broad constructor fragment into a synthetic consumer. Admission now
needs real geometry and resource ownership, which that fixture never modeled.

Clean native qualification exposed two additional stale test contracts: the
resident derivative callback's type assertion omitted its precision-request
argument, and the Direct quarter-exchange fixture used unscaled PBE instead of
PBE0. The original PR library also failed the latter final-state validation.
The fixture now uses the same 0.75 semilocal exchange scale in CUDA and the
independent CPU reconstruction, as the existing DF fixture already did. No
production final-state gate was relaxed.

## Decision

Use the existing initial-guess provenance when selecting each solve. A native
cold guess and an absent seed can use the already-admitted capability. Imported
and moved densities remain excluded from automatic admission. Same-owner warm
proof remains ordinary. Explicit opt-in retains its broader external-seed
contract. A callable constexpr policy tests the combinations without CUDA.

The concurrent remote controller-fixture fix in `b964aee3e` is retained: it
models automatic admission's geometry and resource requirements and checks the
eligible, out-of-domain and ledger-bound cases. Real CUDA public endpoints test
the preparation wiring, seed provenance, resource-ledger exclusion and the
Standard/Fast × Direct/DF combinations. Source-string assertions added by #2696
are replaced by executable policy and endpoint contracts.

## Fast and DF boundary

Do not broaden the automatic guard merely by removing precision or fitted
provider exclusions. Fast has mixed proposals and a separate strict closure;
combining its precision epochs with a strict incremental anchor requires an
explicit transition/rebase policy and independent accuracy/work qualification.
The existing Fast anchor controller has its own narrower nonlocal admission.

DF uses fitted contractions and an occupied-factor exchange path. A signed
density difference does not generally have an occupied positive density factor.
Recomputing dense DF delta contractions also does not inherit Direct quartet
screening savings and can increase work compared with occupied full-density K.
A fitted proposal optimization must establish its own producer contract and
complete-endpoint benefit. Compatibility coverage is not acceleration evidence.

## Invariants

- Native cold provenance can only come from the native initial-guess owner.
- Automatic domain, prepared screening, explicit selector validation, final
  physical/canonical gates, and analytic forces keep their existing contracts.
- Warm, external/moved, ledger-bound, Fast and DF paths retain their owners.
- Numerical references never provide a production density or timed oracle work.

## Evidence

The reproducible endpoint driver is `benchmarks/ks_incremental_default.py`.
It uses separate full-density GPU4PySCF reference processes, the same scientific
inputs and DF auxiliary basis, complete construction/preparation/E/F cold time,
frozen-density warm samples, and independent moved-geometry force checks.
Gates remain 2.5e-6 Eh/atom, 2.5e-5 maximum and 7.5e-6 RMS Eh/Bohr. These are
sampled default-grid solver gates, not quadrature-convergence claims.

Qualification receipts and source/build identity are retained under
`n1:/data/jzzeng/pr2696-fast-df-20261023/` and the qualification worktree's ignored
`.artifacts/` directory. All real-device execution uses finite Slurm allocations.

The [compact qualification](../../../../benchmarks/results/ks-incremental-default-20261023/README.md)
records 44 fresh native processes and 210 complete E/F endpoints, all passing.
PBE0 Standard/Direct cold median unset/off ratios are 0.97191, 0.87288 and
0.88340 at 12/48/96 atoms; warm ratios are 0.99524, 0.99845 and 1.00713.
Three paired alternating-order rounds retain all samples. Cold uses six deltas
within 10/9/9 Focks; frozen warm stays one full Fock. Maximum energy/atom, force
maximum and force RMS errors are 1.3793e-10, 1.1372e-5 and 3.0045e-6 in atomic
units. Fast/DF lifecycle coverage at all three sizes passes without admitting
the Direct incremental controller. B3LYP and M06-2X lifecycle cases also pass.

The CUDA Python suites pass 21 tests. The host qualification passes 339 tests
(24 skipped); the final integrated controller/policy/inventory rerun passes
141 tests. Native physical-state/restart, DF warm-factor, Fast anchor,
composition, SPD values and mixed/strict source work selections pass.

The generated J/K census originally treated automatic MD-J substitution as
lost generated J work. Its fixture now selects generated J only inside that
census and restores the previous environment on scope exit. Other tests retain
default MD-J selection and numerical coverage. A global diagnostic override was
rejected because it changes unrelated coverage; its broad provider run also
exceeded a five-minute allocation. That timeout remains visible and is not
reported as a full provider-suite pass. The focused final SPD and mixed-census
selections pass with the source-matched core.

## Revisit when

Fast has an audited strict-anchor/precision-epoch composition, or DF has a
signed-delta producer with demonstrated semantic-work and endpoint savings.
Neither change follows from algebraic J/K linearity alone.
