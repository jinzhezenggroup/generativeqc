# Bounded stationary capacity for full 96-atom def2-TZVPD

Status: proposed; capacity and geometry gates pass, complete large endpoints pending
Date: 2026-10-04

## Problem

The requested HF-style series is 3/6/12/24/48/96 atoms with full spherical
def2-TZVPD, cold and warm energy plus analytic forces, and displaced geometry.
The last case has 1856 AOs, 1184 packed primitives, 768 shells, and 2359296
points on the matched 48 x 16 x 32 per-atom grid. Its three 1024-AO force
admission barriers are unrelated to the existing through-f integral coverage.
Small accepted endpoints do not qualify the larger domain or its performance.

## Decision and resource audit

Extend the compiler and native stationary allocation bound to 2048 AOs; the
composite driver uses the compiler's existing shape validator instead of a
third independent literal. Preserve the 128-atom/16384-primitive bounds, the
4096-point tile bound, and the requirement for native integrals beyond the
small AO-descriptor diagnostic domain. No enlarged AO^4 descriptor path is
admitted. D/W arrays and AO metadata are dynamic; generated geometry uses
size_t local/global AO indices. Native grid allocation uses checked products,
and its GEMM dimensions remain below INT_MAX. Neither the one-electron provider
nor prepared Direct binding has a 1024-AO array.

The existing exact arena formula, conservative native reserve, optional cache
fallback, and complete host/device admission stay unchanged. At 1856 AOs the
native reserve alone is 885850112 bytes. The native dry nonlocal query for the
full grid and a 256-point pair tile returns 434257940 bytes. With seven runtime
sources and a 1024-point AO tile, the complete planner bounds are 3263916916
device bytes and 3965317140 host bytes before the optional 64 MiB AO-map cache.
These are inventory bounds, not measured peaks. The pre-existing SCF owner and
driver/compiler objects are excluded and must still be provisioned separately.

Keep the production 1 GiB device / 2 GiB host incremental allowance unchanged.
The comparator accepts explicit `--force-max-device-bytes` and
`--force-max-host-bytes`, records both in `native_experiment`, and applies them
only to the scoped WB97M-V force consumer. Four GiB per domain admits this
inventory without reducing the grid or replacing the full basis. Reducing
tile size does not eliminate the native reserve or full-grid nonlocal storage.

## Validation and evidence boundaries

Host tests compile the actual native allocation/create bodies, compare exact
compiler/native bytes through 2048 AOs, check the 1856-AO last density address,
reject 2049 and SIZE_MAX, and reject a one-byte-under mandatory capacity before
allocation. Planner tests retain optional center-cache fallback and full-grid
storage. The real-device geometry gate uses the unmodified 96-atom snapshot,
both spins, dense and noncontiguous maps across index 1024, and independent
PySCF/Libxc finite differences. This isolated slice is not complete-endpoint
qualification: retain separate full cold/warm/displaced E/F gates.

The frozen pre-extension 12-atom campaign (Slurm 5721) now completes all 36
reference/no-source/LDA original/displaced E/F calls. Native/reference warm is
62.961973/18.200958 s and complete cold is 697.619542/181.337490 s. One native
warm includes 33.16 s integral derivatives and 3.31 s combined geometry/pair
drain; the latter is not a VV10-only measurement. LDA reduces target iterations
25 to 19 but costs 262.989625 s and increases complete cold to 803.081423 s.
Its source solve alone costs 262.663043 s. Preserve the no-source baseline;
neither the small-size speed advantage nor the SVP seed gain transfers here.
Raising capacity is not a speedup.

## Qualification receipt

Production commit `302039f8e` builds with 458 verified ccache compiler commands;
208 focused host tests and all hooks pass. Source identity is
`bd67b5aa0b87fc329d84a002c6fae8c518f41177246a5b9d06415b2897810178`;
library SHA-256 is
`4da2d760e4b85e6c8ab5aee199cd541f3240656967f706c0765bdebf62ba488e`.
n5 RTX 5090 Slurm 1439 passes three native programs, four admission cases, and
seven complete E/F/displaced/stale-state cases in each of sparse and zero-cache
modes. Each mode records 66 successful calls and 467 XC submissions. Its job
exit remains 1 because the subsequent new geometry fixture failed.

Slurm 1445 separately passes all four 1856-AO geometry cases (both spins,
dense and noncontiguous AO maps), with unchanged derivative tolerances.
The qualification archive at `296344a5f` differs from the build archive only
in the geometry test file; every other archived file is byte-identical.
The independent read-only receipt is
`.artifacts/tzvpd-capacity96-20261004/capacity-qualification-verified.json`.
It binds both jobs, both archives, the loaded library, scripts and all retries.

Retain the failed fixture attempts: 1439 loaded CuPy's older cuSOLVER before
the native library; 1441 accidentally requested SPD-only integral descriptors
for a geometry-only f-shell test; 1442 retained the small fixture's 256 MiB AO
budget. The correction loads the native CUDA stack first, uses descriptor-free
geometry exactly as the production consumer does, and admits dense global
matrix storage under an explicit allowance. None relaxes native bounds.

Slurm 1443 exposed a separate **oracle representation** mismatch: PySCF stably
groups shells by angular momentum, whereas the external native snapshot retains
its appended diffuse s/p/d shells after f. A shared random density must be
permuted on both indices. The final test derives the permutation from shell
metadata and independently checks all AO jets through order two against PySCF
at 3e-12 absolute/relative gates before using it. No force-fitted permutation or
relaxed finite-difference threshold is allowed. Independently converged endpoint
comparisons do not share a density and were unaffected.

Full 24/48/96-atom campaigns are now running with explicit 4 GiB additional
force allowances. The 48/96 reference and native processes share one allocation
so a multi-GPU scheduler cannot put them on different physical boards. Initial
n4 deployment failed before the engine because NVML and the loaded kernel driver
disagreed; its record is retained and 24 atoms moved to the local RTX 5090.
No complete large-point performance claim is made by this capacity receipt.

## Rejected shortcuts

Do not strip diffuse/f functions, thin the grid, loosen gates, report missing
sizes as zero, lower the conservative reserve without provider evidence, or
interpret task/AO counts as FLOPs. Do not mix the frozen small-size binary with
the capacity extension without a source-equivalence or new qualification record.

## References

- PR #1761; benchmark controls: [same-basis cold note](2026-10-04-tzvpd-cold-density-controls.md).
- Supersedes the admission ceiling, subject to independent resource/numerical
  gates, in [through-f scheduling](../implemented/performance/2026-10-01-default-screened-through-f.md).
