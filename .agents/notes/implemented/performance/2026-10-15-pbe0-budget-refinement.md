# Decision: terminate PBE0 Fast refinement at its authenticated budget gates

Status: implemented
Date: 2026-10-15

## Problem

The precision-bound resident source fixed repeated FP32 recurrence work, but
the 96-atom endpoint still used twelve mixed and five strict Focks. Its final
force error was about 4.16e-7 Eh/bohr against an allowance of 1e-4. The default
density/stationarity tolerance of 1e-5 and a mandatory corrective fixed-point
sweep retained more strict SCF work than the admitted force budget needed.

## Decision

Raise only the admitted budget-Fock default density/stationarity ceiling to
1e-4, matching the native owner's already bounded authenticated stationarity
allowance. Preserve energy tolerance 1e-6, screening 1e-10, the exact scientific
model and every explicitly stricter caller tolerance. Admission still requires
the original energy/force budget floors and native capability/model/grid checks.

For this ceiling only, stop SCF after consecutive fresh FP64 physical Focks pass
the selected physical convergence gates. Do not force another fixed-point SCF
sweep before the separately authenticated final-state validator. That validator
still diagonalizes the current physical Fock and checks density reconstruction,
stationarity, trace, idempotency and canonicality. Its numerical/algebraic gates
are not bypassed; a rejected state cannot supply forces. Stricter density
controls retain the incumbent corrective closure, as do non-budget consumers.

This is not a fixed two-iteration refinement schedule. Two fresh strict Focks
are the measured successful minimum on this case; difficult states retain
bounded correction/fallback. Forces and final physical Fock arithmetic remain
FP64. No approximate energy is accepted as the strict energy-history baseline.

## Rejected alternatives

- Removing strict physical evaluation outright is not supported: the original
  mixed energy at iteration twelve differs from the independent reference by
  about 2.2e-4 Eh, exceeding the 1e-5 total-energy allowance.
- Combining FP32 J/K with the existing FP32 PBE point/AO warmup and FP32 tiled
  inner sums triggered a numerical retry on the first proposal. Retain that
  attempt and its work; do not present a mostly strict retry as successful
  mixed-path acceleration.
- Retaining FP64 inner sums while combining the PBE point/AO warmup still
  failed complete force final-state validation. This happened with both the
  previous and the larger stationarity allowance. The full endpoint fell back
  to strict solves. These candidates are not enabled or included in successful
  speedup claims. Do not weaken trace/idempotency/canonicality to admit them.

## Evidence

Source-matched local evidence, including rejected candidates, is retained in
`../evidence-pbe0-budget-xc-20261015/`. Initial screening of the selected approach
in finite Slurm job 8510 gives one complete energy/force endpoint in 62.251794 s:
twelve mixed plus two strict Focks, one solve, zero numerical retries. Against
the same scientific-protocol independent GPU4PySCF oracle, total-energy error
is 2.28702e-8 Eh, maximum force error 1.05285e-5 Eh/bohr and force RMS error
3.96049e-6 Eh/bohr, within the original 1e-5 / 1e-4 / 3e-5 observable gates.
This single screening time is not the repeated paired performance conclusion.

Host tests cover default and caller-capped controls, explicit final-state
allowances, stricter density ceilings and unchanged determinant/canonicality
rejections. Final qualification must preserve independent-process optimized
FP64 residency, complete forces/export/teardown, actual semantic work and
source/build identities; do not divide timings from different GPU allocations.

### Repeated complete-endpoint qualification

Finite Slurm job 8513 alternates six independent processes on one RTX 5090
(`GPU-45f177f9-33ac-0e4e-f473-2beeaf6d413d`), three per mode. Every FP64
resident source and both Fast phase sources are accepted; no comparator
optimization is disabled. All six endpoints pass independent energy/force
gates. Construction, SCF, complete moving-grid forces, export and teardown are
included, with no unreported warmup process removed from the comparison.

| Mode | Three complete endpoint times (s) | Median (s) |
| --- | --- | ---: |
| Optimized FP64 | 77.297521, 79.378250, 78.629746 | 78.629746 |
| Fast | 62.140407, 62.521923, 62.364804 | 62.364804 |

The paired comparison gives **1.260803x**, or **20.6855% less wall time**.
Every Fast endpoint has twelve mixed plus two strict Focks, zero numerical
retries and no post-SCF Fock builds. Maximum errors across these Fast samples
are energy 2.28702e-8 Eh, maximum force 1.05285e-5 Eh/bohr and force RMS
3.96049e-6 Eh/bohr. These are case-specific observed errors, not a new universal
error certificate. The earlier resident-source campaign used another physical
GPU; do not divide its Fast wall time by this campaign's wall time and call
that a paired revision-to-revision speedup.

Source-matched small water and spherical-tetramer endpoints also pass the
independent gates, but remain slower than FP64 (single samples: 3.392465 versus
1.572867 s, and 4.957184 versus 3.010513 s). An explicit 1e-5 water density
tolerance is preserved and passes the stricter retained closure path. Finite
Slurm job 8516 checks the complete default Fast water endpoint under Compute
Sanitizer memcheck: zero errors. The final host regression selection passes
290 tests; the analytic native final-state executable passes separately.

The measured source/build identity is
`02d03749a75be8b3af2fbcc9a57f10d4787c4f7e13f349a4336e6ebfce99b59f`;
the Release/sm120/CUDA-12.9.1 normal-compile main-library SHA256 is
`6bac6c2e050de0513ac12a63d7e94ee1bd6361f6efc3240f820aae9b3d315cb0`.
The unchanged scientific-protocol independent oracle SHA256 is
`eef5fec85249437ef136af54149b2becfecbf39edce715dd7d48d723115f9c15`.
The full deployed input inventory has 11,010 verified files. Raw records,
source admission traces, masks, source archive/patches, sanitizer output and
`comparison-final.json` are retained in the evidence directory above. Later
note-only result appends do not relabel the measured production source.

Before submission, the existing PR's pre-commit formatting commit
`efe0db1b435374c263f7c133cda0a05e5315b2af` was preserved by fast-forwarding
the branch and reapplying this refinement. Its eight native files have identical
clang-format-normalized contents, and its Python test has an identical AST.
After integration, 118 policy/component host tests and the analytic native
final-state executable pass; Ruff and changed-line clang-format checks pass.
This is not a byte-identical final-head GPU rerun: the measured archive and
binary identities above remain the authority for the GPU numbers.

## Revisit when

Combined XC lowering passes the complete relaxed-density force proof and
independent observable gates, or a cheaper physical energy evaluation replaces
the strict phase without accepting the known mixed-energy bias. Optional
FP64/source residency must not be disabled to improve the comparison.

## References

- [Earlier precision-bound resident-source decision](2026-10-15-pbe0-mixed-resident-source.md)
- [Original budget-Fock numerical decision](../numerics/2026-10-14-pbe0-budget-fock.md)
- `docs/user/application_precision.md`
- `benchmarks/pbe0_fast_fock.py`
