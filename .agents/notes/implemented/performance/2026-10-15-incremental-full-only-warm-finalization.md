# Decision: reuse full-only resident warm HF states with incremental J/K enabled

Status: implemented; incremental J/K remains experimental and off by default
Date: 2026-10-15

## Problem

The fixed-anchor delta-D experiment exposed a separate finalization cost. A
resident warm RHF solve that converges without a delta build still executes two
full physical builds solely because incremental J/K was requested. The ordinary
HF path already has a qualified same-generation force-state reuse contract.
The pre-repair generic-build measurements show 68--80% warm wall-time regressions,
despite cold/moved wins. Those measurements and their qualification limits remain
in the [fixed-anchor note](../numerics/2026-10-15-incremental-fixed-full-anchor.md).

## Decision

Keep the incremental convergence graph and strict delta finalization unchanged.
Track determinant orbitals in the already allocated weighted-density scratch
whenever the plan permits ordinary force-state reuse. After SCF, a resident
warm force bucket may enter that reuse path only when:

- the public endpoint requests forces, not detached reference export;
- every seed is warm and has the existing exact resident density/geometry/plan
  provenance, including the coefficient frame from its previous publication;
- ordinary converged-state reuse is enabled, with no explicit final-rebuild
  diagnostic; and
- every item's actual incremental delta-build count is zero.

Read the tiny delta-count vector through the existing final-Fock decision
residency boundary, with the explicit `incremental_delta_count` payload. Reuse
that host snapshot for result diagnostics rather than downloading it twice.
There is no additional quartet enumeration or GPU matrix allocation.

Incremental convergence normally accepts `P_(n+1)`. With no deltas, the last
full anchor still contains the preceding `P_n`, paired with the retained raw
physical `F(P_n)`, energy and pre-DIIS stationarity check. Restore that density
for successful items and publish its tracked `C_n` orbital frame through the
existing force-ready finalizer. This reproduces ordinary retained-generation
semantics without changing the incremental graph or treating a screened delta
Fock as an exact physical Fock.

## Invariants and rejected alternatives

- Do not simply remove the blanket reuse prohibition: that would permit a
  screened reconstructed Fock to escape required strict finalization.
- Do not reuse `F(P_n)` alongside the already promoted `P_(n+1)` density or its
  trial orbital frame; energy, forces, Pulay and restart must share one state.
- Check actual delta work, not `iterations == 1` or merely the requested option.
- Whole-bucket admission is deliberate. Any delta, partial/unproven warm seed,
  moved geometry, reference export or explicit final-rebuild selection keeps
  the bounded canonical fallback. Per-item mixed reuse is not claimed.
- Physical convergence gates, failed-state cache invalidation, strict FP64
  admission, source ownership and default selectors are unchanged.

## Validation contract

Native CUDA RHF and UHF tests exercise repeated resident replays, exact retained
density generation, published force-state audit, independent pinned PySCF
energy/force gates, external seeds, moved geometry and forced full finalization.
Existing cold incremental tests retain their strict physical-build gates.
Actual device execution and repeated complete endpoint measurements require
finite Slurm allocations; production-AOT/KS/default qualification remains
separate from this narrowly bounded repair.

## Matched-source validation

The repair is built on master
`4e28477e96d257528a60bcb68c00529f22585bc1` (including the subsequent guarded DF warm-factor default, not just
the earlier fixed-anchor experiment's master). The Release/sm120 CUDA 12.9.1
build uses the scalar CPU linear algebra provider, generic/non-AOT force paths
and existing ccache compiler launchers. It is a correctness/diagnostic build,
not the production performance profile. Its library SHA256 is
`3ccfcbf513bca1496a5a5737b2ab80f570c1058625a69e193d69d6faf65001d2`;
native source-content identity is
`743819d554a2b4c43e6a98c3d1ed56147c53b78c84798390e3f1cfb562b3b5dc`.

- The focused host suite passes 330 tests, with nine explicitly GPU-gated
  skips. CPU Fock-provider raw matrices, energy, metric response and preflight
  tests pass. Nonstandard library/dependency search paths must be supplied;
  initial missing-library/OpenSSL/LibArchive setup failures are not numerical
  results or changes to those tests.
- Slurm job 8773 on n1 passes native incremental RHF/UHF, interval 0/1/8,
  pinned independent PySCF E/F, repeated warm replay/audit/generation, external
  seed, moved-geometry and explicit-rebuild gates. Warm RHF/UHF each perform
  one full SCF build, zero deltas and zero final full builds; selecting
  `GENERATIVEQC_FINAL_FOCK_REBUILD=1` retains the two final full builds on
  the same no-delta cases. Native KS incremental physical-state and fresh-solve
  tests also pass. Independent live CPU PySCF water/OH E/F checks and the
  two-geometry frozen-warm OH force-state regression pass with incremental
  execution both off and on.
- Slurm job 8765 on n1 runs the newly built, actual multi-CTA controller
  probe under memcheck and synccheck, both with zero errors. Slurm job 8776
  runs the complete native incremental HF (including new warm cases) and KS
  executables under memcheck, both with zero errors.
- This does not clear the full HF synccheck blocker recorded in the
  fixed-anchor note: the unchanged initial full provider reported 96 divergent
  warp barriers before delta execution. No complete synccheck pass is claimed.
  An attempted n2 sanitizer allocation failed NVML driver/library preflight;
  the accepted molecular sanitizer run is on n1, not n2.

## Complete endpoint observations and negative large-case evidence

The retained [source-matched compact record](../../../../benchmarks/results/incremental-full-only-warm-20261015/README.md)
contains per-endpoint timing/work/gates, build/allocation identities and hashes.
The full references/force arrays, reconstruction patch, validation outputs and
frozen reproduction tools remain in ignored local
`.artifacts/delta-anchor/warm-full-reviewed-bundle`. The repository aggregate
evidence budget has insufficient space for that full bundle; no older evidence
or retention policy is changed. Its local availability is not a permanent
remote-archive or standalone-reproduction guarantee.
Slurm job 8775 on n1 interleaves OFF and fixed8 using the same library and
assigned RTX 5090, independent prepared owners and their own cold-derived warm
densities. Six repeats per arm balance the projected arm order. Both arms
receive an unmeasured complete priming trial. No process-bootstrap advantage or
imported reference density is counted as a win.

Median complete RHF/def2-SVP spherical FP64 energy-plus-force times, seconds:

| AOs | Phase | OFF | fixed8 | Wall-time change |
| ---: | --- | ---: | ---: | ---: |
| 96 | owner-cold | 1.447577 | 1.274201 | -11.98% |
| 96 | warm | 0.187664 | 0.186917 | -0.40% |
| 96 | moved | 1.151983 | 0.963352 | -16.37% |
| 192 | owner-cold | 13.556668 | 11.716461 | -13.57% |
| 192 | warm | 1.954380 | 1.959842 | +0.28% |
| 192 | moved | 11.474454 | 8.561851 | -25.38% |

Warm now has one full SCF build, zero deltas and zero final full builds in
every sample. Its timing is approximately baseline parity, not a significant
speedup over OFF. All paired cold/moved samples favor fixed8, but these remain
generic-build observations, not production qualification. In particular,
192-AO cold requires 17 rather than OFF's 16 iterations. All 72 formal and
12 priming endpoints pass 1e-8 Eh / 1e-7 Eh/Bohr independent CPU PySCF gates;
maximum errors are 8.413e-12 Eh / 9.325e-11 Eh/Bohr. Owner-cold charges
construction, preparation, initialization, SCF, physical finalization and host
E/F publication, excluding common process/library/CUDA bootstrap; teardown is
separate. Quartet work is known at 96 AOs but unavailable for the bounded
streaming 192-AO provider and is never reported as zero.

The separate 384-AO n4 experiment, Slurm job 992, is not accepted. Fixed8's
moved priming point exhausts 100 returned iterations (12 full, 88 delta), plus
a discarded 100-iteration target attempt. Density RMS is 7.825e-11 and energy
error is 1.000e-11 Eh, but the solve is not converged, returns no forces and
reports fallback. Its small density step is not proof of a physical endpoint,
and does not establish the cause of the nonconvergence. The remaining cohort
is explicitly cancelled after observing that rejection; six priming and five
completed formal records, the failure and cancellation remain in the bundle.
Do not select successful partial phases to claim a 384-AO speedup or relabel
the historical pre-repair large-case success as validation of this repair.

This repair is deliberately limited to trustworthy no-delta warm publication.
It does not solve every screened-delta convergence trajectory. Further large
HF investigation must distinguish full/delta energy oscillation, stationarity
and retry behavior, with independent E/F gates and failed points retained;
do not weaken tolerances or silently promote a full-energy-refinement policy
to make these measurements pass. KS full-energy refinement remains unchanged.

## References

- `src/scf/cuda_rhf.cpp`
- `tests/native/test_incremental_direct_jk_cuda.cpp`
- [Current HF architecture](../../../../docs/developer/architecture.md)
