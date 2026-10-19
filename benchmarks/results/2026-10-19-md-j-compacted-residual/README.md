# Compacted exact-mask residual J: retained qualification

This is historical evidence for the opt-in `(2,1)` compacted residual consumer,
not default-promotion evidence or a claim of GPU4PySCF-equivalent performance.
The implementation is stacked on PR #2600, commit
`aacbccc275274cb70e36b76a646750fe667acf3e`. The native source and compiler module
SHA256 inventories in `summary.json` match the submitted implementation.
This packaging deliberately excludes the later primitive-strip prototype.

## Scope and measurements

The prepared-owner endpoint starts with host canonical density and finishes
with synchronized host canonical J. Setup and a fresh independent CPU libcint
reference are outside the timing. Native timing includes the test adapter's
input-density readback/assertion. Each allocation has four interleaved,
reversed-order samples per method, on one Slurm-assigned RTX 5090.

| Slurm allocation | Native incumbent | Compacted `(2,1)` | GPU4PySCF 1.8.1 default | Native time reduction |
| --- | ---: | ---: | ---: | ---: |
| 10353 | 4.026204 s | 3.717649 s | 0.690771 s | 7.66% |
| 10354 | 3.883460 s | 3.577326 s | 0.670088 s | 7.88% |

Inputs are 96 atoms, 1088 spherical AOs, the explicit canonical r2SCAN-3c basis
and the same frozen physical density. Their identities, native library/source
identities and installed reference identities are retained in `summary.json`.
Native maximum J error is about `2.456e-11` against the fresh CPU reference;
candidate/incumbent differences are at most `3.233e-13`. The independent J gate
is `atol=3e-9, rtol=3e-11`.

Native uses its original `1e-12` directional FP64 AO screening and global
density-error allowance; GPU4PySCF's default is `1e-13` with a different
screening budget and hierarchy. Their screening domains are **not identical**.
The comparison does not attribute the remaining approximately 5.3-fold timing
gap to one kernel or one hardware bottleneck.

A separate, single paired cold SCF plus host-force endpoint (Slurm 10355)
changes from 271.237524 to 263.289601 s including preparation, 2.93% shorter.
Both arms take 24 SCF iterations and 24 Fock builds, use freshly rebuilt
source-matched stationary AOT, and pass independent energy/force gates.
Maximum candidate errors are `2.456e-11` Ha and `5.844e-11` Ha/Bohr.
This is not a repeated endpoint distribution or a GPU4PySCF full-SCF comparison.

## Qualification and unchanged contracts

- PR packaging: 110 focused host tests pass, including the parent compact-E
  capacity and diagnostic contracts; compiler structure checks 568 modules
  with zero dependency errors, and Ruff lint passes.
- Slurm 10352: 76 real-CUDA oracle cases ordinarily and under each of memcheck,
  racecheck and synccheck; no reported errors or hazards.
- Slurm 10353: 114 normal molecular cases in each of modes 0 and 2.
- Slurm 10357: all 25 native scientific work domains agree, including uniform
  sources/directions/summands and residual tasks/primitive products. Tested plus
  coarsely screened candidates conserve the original search domain.
- Slurm 10356: fresh authorized Nsight Compute evidence is described in the
  decision note. Static resources and theoretical local request sectors are not
  treated as proof of actual DRAM traffic or a unique bottleneck.

Select the measured candidate with `GENERATIVEQC_MD_J_MASKED_TILES=2`, leaving
`GENERATIVEQC_MD_J_MASKED_RESIDUAL=0` and
`GENERATIVEQC_MD_J_MASKED_PREPROJECT=0`. The new selector remains off by default.
`(2,2)` retains its incumbent path. Existing parent reciprocal and coarse-tile
defaults are unchanged. Exact directional masks, UKS/nonsymmetric density,
orbit exclusion, precision, the original optional 128-MiB MD allowance and
bounded fallbacks remain authoritative. No production CPU oracle is introduced.

The PR is a draft: the parent is not merged, larger-size/fallback qualification
and repeated complete-endpoint cohorts remain necessary before promotion.
Changes to the native or compiler source invalidate transfer of these receipts
unless independently requalified. No new GPU timing was run merely to package
the already-qualified source into a PR.

## Receipts

`summary.json` retains both prepared-J cohorts, per-class scientific work and
the cold endpoint summary. Full logs, reference matrices, compiler commands,
cache statistics and profiler reports remain in the ignored local
`build/warp-coarse-final/` and immutable remote qualification directory
`/home/jzzeng/generativeqc-qualification/warp-coarse-final-20261019-49fc`.

The [decision note](../../../.agents/notes/implemented/performance/2026-10-19-warp-coarse-masked-hermite.md)
records the algorithm, superseded variants, fallback choices and promotion
boundaries. The in-repository host and real-CUDA tests are
`tests/python/test_md_j_masked_codegen.py`,
`tests/python/test_md_j_masked_cuda.py`, `tests/python/test_md_j_resources.py`,
`tests/python/test_md_source_launch_status.py` and
`tests/python/test_md_j_normal_cuda.py`. All actual GPU validation must use a
finite `srun` allocation and preserve its assigned device visibility.
