# Rejected: scaled Becke products for radius-adjusted moving grids

Status: rejected; measured cold gain below noise, production guard restored
Date: 2026-11-09

## Problem

The new-master DFT-only PR #3051 qualifies shared-Rys streaming, but B3LYP
Direct96 still trails the same-allocation GPU4PySCF cold by 0.168 s (1.44%).
Its fresh A100 cold census attributes about 0.85 s to four ordinary Becke
pair/product/reverse/gather phases. The production radius-adjusted grid cannot
use the existing binary-scaled representation because `adjusted_partition`
explicitly excludes it, despite both representations sharing coordinate AD.

## Candidate and invariants

Remove that exclusion only. Retain automatic 96–128-atom admission, the existing
phased reservation, actual native capability checks, optional-allocation
fallback, center-geometry lifetime and primitive-adjoint scratch-alias refusal.
Small automatic owners keep ordinary logarithms; the existing explicit control
continues to compare representations. No GPU model/SM profitability rule,
additional allocation, point cutoff, approximation or production CPU work is
introduced. First derivatives alone change; SCF equations/convergence and
second derivatives are outside this representation's existing domain.

The shared Treutler-adjusted coordinate AD remains the mathematical owner.
Scaled primal stores both corrected coordinate partials alongside the canonical
switch factor/slope. Reverse consumes those stored partials rather than
reconstructing an equal-radius coordinate. Elemental radii are immutable
metadata, not geometry variables. Ordered scaled products replace logarithms
without altering the point/atom order, four-word pair panel, seven phases,
geometric gather, exact-zero or single-zero excluded-product semantics.

## Predeclared acceptance

Require complete cold energy, analytic force and host export, fresh-process
control/candidate/candidate/control for PBE0/B3LYP Direct96, all three own-density
warms individually paired, independent dense/grid-stability/CPU-GPU spot and
strict native E/F gates, unchanged solver/settings/initial-guess work and real
cold-only kernel census. A100 cold gains must clear max(2%, twice the sum of
within-arm relative MADs); every paired warm and 5090 cold must stay within 3%.
Retain Direct48 fallback controls and additional r2SCAN-3c/ωB97M-V paired
diagnostics without calling single pairs balanced qualification. Preserve all
failed observations. Do not reuse old-image timings or enlarge the 85 MiB cap.

## Measured evidence and rejection

- All 2244 focused host tests pass. New adjusted tests cover direct/prepared
  center geometry, 2/5/48/96/128 atoms, clipped heteronuclear radius ratios,
  both pair orientations, signed seeds and moved/restored centers. Independent
  80-digit direct-product Decimal differences, translation and atom/radius
  permutation pass for iterations 1/3/5 and validated/unvalidated schedules.
- Normal dual-target Release build Slurm16865 and actual shared-owner fixture
  build16866 exit0 with verified sccache. Full image source9d11d755 / ELFf44e143a
  is distinct from PR #3051's qualified source75d94b32 / ELF549ed9a2.
- First fixture allocation16867 fails before endpoints due solely to the newly
  generalized test handle being shadowed by its existing buffer dictionary.
  Preserve that failure. Rename the test handle without rebuilding production.
  Corrected allocation16868 passes all36 actual native-owner cases, including
  radius-adjusted automatic selection and bounded/alias/invalid-publication
  fallbacks. Its complete endpoint population, four cold profiles and
  memcheck/racecheck/synccheck finish successfully with zero errors/hazards.
- A100 CPU-only LSF230463 authenticates the actual gpu22 base image, not GPU
  performance. Exact delta transport locally reconstructs the complete original
  1,164,879,152-byte candidate ELF; no source input or executable body is pruned.

| Card | Complete cold Direct96 | Control | Candidate | Change |
| --- | --- | ---: | ---: | ---: |
| A100 | PBE0 | 11.751200 s | 11.686492 s | -0.551% |
| A100 | B3LYP | 11.848846 s | 11.796362 s | -0.443% |
| 5090 | PBE0 | 16.814291 s | 16.525521 s | -1.717% |
| 5090 | B3LYP | 16.877596 s | 16.539189 s | -2.005% |

Both arms use the same experimental source9d11d755 / ELFf44e143a, with explicit
disable versus automatic selection. They must not be relabeled as executions of
the restored source75d94b32 / ELF549ed9a2. A100 fails the 2%/noise gate. Even
without its 2% floor, gains0.551%/0.443% are smaller than twice the sum of
within-arm relative MADs0.658%/0.566%. All primary paired warms pass; 5090 warms
gain7–8%. A 5090 Direct48 single pair regresses3.618%, despite unchanged automatic
selection. This neither establishes causation nor qualifies nonregression.

A100 LSF230464 passes36 native tests and finishes all18 noninstrumented
populations, each cold plus three own-density warms. It then exits1 before the
first profile: `libcupti.so.12` cannot load. The older worker created a symlink
to retained `libcupti.so.12.9`; this worker omitted it. All four A100 profiles
are missing. Do not claim actual A100 production dispatch or retry successful
populations to rescue a candidate already rejected on cold performance.
5090's four changed kernels execute1054 times each, and all seven Becke force
phases retain that count; the once-only index setup is not an eighth force phase.

All present independent dense/grid-stability/CPU-GPU spot and strict native
E/F/work gates pass. Primary max energy/force differences are about
1.36e-12/4.03e-13; r2SCAN-3c max force difference is about1.53e-11.
The CPU/GPU independent nonlocal-grid comparison matches every configuration
field except implementation-specific retained point count (90064 versus90112
at identical level6/NLC5); no oracle is rewritten or tolerance weakened.

Two fresh A100 reference colds per method give means11.064521s PBE0 and
11.615196s B3LYP. The experimental incumbent-control means remain slower by
0.686679s/6.206% and0.233650s/2.012%. Historical PBE0 reference anomalies
165.394061/30.152892s retain their original observations and unknown cause.
Do not claim universal A100/README leadership.

## Decision and revisit conditions

Restore `!adjusted_partition` verbatim. Keep PR #3051's independently qualified
streaming runtime unchanged. Retain the host radius-aware oracle tests, generic
actual-device fixture, reusable independent scientific gate, original experiment
and explicit rejecting replay. Do not enable only on5090 to harvest warm gains:
the task is A100 complete cold, not a model-specific warm optimization.

Revisit only with a materially different algorithm/work reduction and a
predeclared complete-cold gain comfortably above noise, complete A100 collector
provenance and physical5090 nonregression. Next measured targets are the nine
SCF value-Fock launches of DPPP about0.98s and PPPP about0.71–0.74s, separately
from the once-only spatial-SPD force owner about0.69s. They are not all
derivative kernels; preserve the census names/counts when selecting a source.

## Rejected shortcuts and references

Do not revive the rejected multi-gigabyte AO cache, weaken scientific/solver
gates, infer endpoint savings by subtracting kernel sums, or silently add this
unqualified source to the already ready/CI-green PR #3051. See #2976 and
`../implemented/performance/2026-11-09-dft-streaming-master-qualification.md`;
the public capsule is `benchmarks/results/dft-adjusted-becke-20261109/`, and the
ignored experiment root is `.artifacts/dft-adjusted-becke-20261109/`.

The existing evidence cap initially rejected the added capsule by180573bytes.
Keep the cap unchanged: repack only the CO2/TZVP and water-tetramer/SVP input
NPZ containers using stdlib ZIP-LZMA, already used by the water-octamer fixture.
Save266242bytes while proving every original ZIP member name and NPY byte
unchanged. Update only compressed-container hashes/sizes; numerical metadata,
array dtype/shape/hash identities and all reference consumers stay unchanged.
The capsule's `input-storage-receipt.json` binds original and new containers and
every NPY member. Other measured recompression trials are not applied.
