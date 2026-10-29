# Decision: retain specialized RSH derivatives across resident source transport

Status: implemented
Date: 2026-10-30

## Problem

The resident Direct RSH force adapter always selected the generic fused radial
traversal. The existing host adapter already decomposed the canonical omega=0.3
case into specialized full-range J/K and long-range exchange. Moving the result
transport to the stationary device owner had lost that scheduling decision.
The stationary receiver also lacked a full-range K destination for RSH profiles.

On an A100, a 12-atom WB97M-V diagnostic attributed 2.486485 seconds to
`direct_rsh_force_resident`, despite one warm SCF iteration. Clean complete warm
energy-plus-force calls took 4.053219 seconds. This was a performance defect in
the resident path, with no numerical discrepancy in the independently checked
endpoints. The diagnostic CUDA-event ledger is retained; Nsight's importer failed
with an event-order error, so it does not establish a kernel census.

## Decision and invariants

For omega=0.3 and a nonzero LR correction with a finite SR/correction ratio,
publish `FullRangeJk` followed by `RangeCorrection`. Both providers use the
existing prepared owner and stream. Views expire when their callbacks return:
the receiver must enqueue the full-range copies before the next traversal
reuses provider storage. No host force array, extra integral owner, new scalar
algebra, GPU-model allowlist, or inherited tuning claim is introduced.

The RSH stationary receiver temporarily puts full K in the short-range slot.
Its existing correction transport subtracts the scaled LR contribution from
that slot and fills the long-range slot. It marks only the short-range source
as present until the correction arrives, preserving incomplete/duplicate source
rejection. Force-to-gradient signs and the model's SR/LR coefficients remain
owned by the same transport.

Other omega values, a zero correction, and nonfinite recomposition ratios retain
the complete fused provider. A failed provider propagates failure; it cannot
publish a successful complete endpoint with only the first source. The omega
boundary describes the existing qualified radial specialization, not a hardware
restriction. Broader admission requires separate radial qualification.

## Rejected alternatives

- Returning to host force arrays would restore unnecessary transfers and weaken
  callback-scoped source ownership.
- Reimplementing LR mathematics or tuning the generic fused traversal would
  duplicate already qualified providers before recovering the lost schedule.
- Promoting partial, unchecked large-system timings would hide missing work or
  oracle gates. Such attempts remain diagnostic records only.

## Qualification

The ECNU campaign uses finite LSF allocation 216451 on gpu22, queue `gpu_chem`,
eight CPU slots and one exclusive-process A100 PCIe 40GB. CUDA visibility is
preserved. Native builds use CUDA 12.9.86, GCC 14.4, Release SM80 cubins and the
portable profile with the existing verified ccache. The reference uses
GPU4PySCF 1.7.6, PySCF 2.13.1 and CuPy 14.1.1 without cuTENSOR on the same
allocated GPU. The task-private glibc/runtime environment does not replace
system libraries. Its native test executable needs a task-local `-rpath-link`
to CUDA libraries and is invoked with the private loader; this is not a
production CMake change.

The endpoint driver retains cold setup plus energy/analytic forces, three warm
calls, a moved geometry and three moved-warm calls. Standard cases use frozen
spherical def2-SVP; r2SCAN-3c retains prescribed def2-mTZVPP, D4 and gCP. PBE
uses its maintained pruned grid; the other RKS cases use 48x16x32 quadrature.
WB97M-V retains LR exchange and VV10. Gates are 1e-8 Eh for energy and 1e-7
Eh/Bohr for maximum force error against independent references.

Matched grids diagnose implementation agreement. They do not establish
grid-converged production accuracy or an overall engine ranking. Reference SCF
counts differ and remain visible; comparisons must not equate an iteration
with an identical amount of scientific work.

Initial clean WB97M-V before/after complete endpoint times (seconds) are:

| Atoms | Phase | Previous resident path | Specialized resident path |
| --- | --- | ---: | ---: |
| 3 | cold | 3.037302 | 2.920843 |
| 3 | warm median | 0.427299 | 0.290524 |
| 3 | moved | 1.716922 | 1.596461 |
| 3 | moved-warm median | 0.431217 | 0.291712 |
| 12 | cold | 13.066112 | 11.102799 |
| 12 | warm median | 4.053219 | 2.095206 |
| 12 | moved | 10.253613 | 8.316820 |
| 12 | moved-warm median | 3.991309 | 2.104016 |

All 16 fixed records pass. Native SCF counts are unchanged between these arms;
each warm call uses one Fock. Raw samples, AO work and nonlocal-work receipts
are retained. These are one fresh process per arm with three warm repeats,
not a broad hardware or startup-variance study.

The compiled host adapter fixture checks RKS/UKS coefficient reconstruction,
zero SR, noncanonical omega, zero/subnormal correction, invalid scientific
identity, provider failures and destructive callback-scoped storage reuse
(42 cases). It is a transport test, not an integral oracle. The actual A100
native binding executable passes all eight built profiles, including RKS/UKS
RSH source recomposition, destructive provider scratch reuse, incomplete and
duplicate correction rejection, arena admission and lifetime checks.

The final source passes 1,000 host checks across the resident adapter, native
stationary binding, shared stationary kernels and Direct RSH callback contracts.
Selected pre-commit checks pass. The baseline used for the RSH timing comparison
already includes the restored portable J/K work, isolating the resident RSH
change from that earlier optimization.

## Provenance and follow-up

Ignored raw evidence is under `.artifacts/ecnu-a100-functionals-20261029/`.
The baseline production inventory is
`b2ae0550b2e95eff39850166b2ad4a2e8e01ba8ba6e4f9df5142ee7605420d83`;
the corrected production inventory is
`e2f28f36363baaa5da97abe7062a32b1ea4d853f5d39b8ff5cecb580986fd53a`.
Binary hashes, source inventories, dependency hashes, inputs, scheduler and
build receipts accompany the endpoints. Failed environment setup and interrupted
unchecked measurements remain separate from qualified complete cohorts.

PR #2880 was merged during qualification, but its master tree omitted the
final portable generated-J/K/cold-reuse changes from the earlier tested branch.
The follow-up restores those unchanged, previously qualified paths as well as
this RSH fix. See [the portable J/K rationale](2026-10-29-portable-rys-jk.md)
for that separate implementation and its complete numerical/performance evidence.

Revisit the canonical boundary when another radial schedule has independent
force qualification. Revisit the transport if provider streams, source lifetimes
or stationary slot ownership change. Any later optimization must retain complete
energy-plus-force gates and disclose SCF, J/K and nonlocal work.
