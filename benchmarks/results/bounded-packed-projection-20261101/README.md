# Bounded packed occupied projection

The default packed DF projection uses bounded unpack plus FP64 BLAS when the
CUDA FP32/FP64 throughput ratio is in (0, 2], nbf >= 256, naux >= 1024,
64 <= occupied rank <= nbf, and charged scratch holds at least eight full-Q AO
rows. Other domains retain generated projection. Explicit `generated` and
`blas` controls remain available on every GPU. No GPU product name, grid,
precision or convergence threshold controls this optimization.

Each panel borrows at most 128 AO rows from the existing exchange-intermediate
scratch, which is dead until the K Gram. U remains disjoint and retains its
final-force lease. If explicit BLAS cannot hold one row, execution retains the
generated kernel without allocating another tensor. Work counters include the
additional unpack traffic and preserve the original projection/K work.

## Complete endpoints

All comparisons below are against **generated GenerativeQC**, not GPU4PySCF.
The protocol is README water96, fitted J/K, spherical def2-SVP, default SCF/XC
controls, analytic forces including grid response, and force tile1024. WB97M-V
retains its separate exact range correction and the eager warm-energy anchor.
Math libraries use one thread; each finite allocation reserves eight CPU slots.

Each method has two separate cold/one-warm conditioning owners, followed by
baseline/candidate/candidate/baseline owners, each with cold plus three frozen
warm energy/force calls. Cold includes construction and preparation. The table
reports medians, including every complete endpoint, with negative changes faster.

| Automatic version | Generated cold (s) | Auto cold (s) | Cold change | Warm change |
| --- | ---: | ---: | ---: | ---: |
| A100 PCIe 40 GB, PBE0, LSF224180 | 12.45552 | 10.96669 | -11.95% | -5.32% |
| A100 PCIe 40 GB, B3LYP, LSF224180 | 12.56924 | 11.08612 | -11.80% | -5.27% |
| A100 PCIe 40 GB, M06-2X, LSF224180 | 14.90337 | 13.24758 | -11.11% | -4.67% |
| A100 PCIe 40 GB, WB97M-V, LSF224181 | 141.09007 | 136.35876 | -3.35% | -0.67% |
| RTX 5090, B3LYP, Slurm n1/15042 | — | — | +0.154% | -0.071% |
| RTX PRO 6000, B3LYP, Slurm n2/3841 | — | — | +0.077% | +0.036% |

The Blackwell automatic domain retains generated projection; these small
variations establish neither a speedup nor a degradation. All paired physical
iteration counts, Fock counts and diagnostic work are unchanged. Every warm
call performs one iteration and one Fock. Maximum paired errors across the
automatic populations are below 1.37e-12 Eh and 1.68e-12 Eh/Bohr.

The earlier explicit-only version is also retained: A100 LSF224175 cold
PBE0/B3LYP/M06-2X changes -11.97%/-11.81%/-11.13%, warm
-5.24%/-5.11%/-4.83%. Those measurements are not relabelled as automatic results.
The reader rechecks all **54 complete populations / 180 energy-and-force calls**
against the unchanged independent README energy, full-force, grid-stability and
CPU/GPU spot gates. This includes conditioning, not only timed ABBA samples.

The native automatic source identity is
`eb69035ae9a9e3586d63c95dc1b5ab1667ddc40d252cbc460637f5ee935e17f4`;
the earlier explicit version is
`9021d2769764d52f975140d7769feb33109d9dd3e6e6dc41208af40bda664be1`.
Original receipts identify their base commit and dirty source. Per-device binary
hashes and physical GPU locators remain in the receipts. CPU provider/build
choices also require their own provenance: native source identity alone does not
identify the linked provider or authorize combining timing populations.

## Negative evidence and validation

The isolated production-kernel probe is diagnostic evidence, not an endpoint
speedup. For n=768/a=3616/rank=160, A100 column-major projection improves from
248.28 ms generated to 79.40 ms with 128 rows; the 5090 changes from 353.86 ms
to **423.66 ms**. Row-major and tail controls are retained. A long-double sample
oracle checks the arithmetic independently. Unconditional BLAS is therefore
rejected. The two earlier A100 loader failures retain their nonzero exits and
supply no timings. Their corrected matching-sysroot-loader run is LSF224169.

The automatic version passes 18 host policy/projection tests and nine GPU cases
on each of A100 (LSF224179), 5090 (Slurm15042) and PRO6000 (Slurm3841).
The GPU cases use independently reconverged PySCF energy/full-force references
and cover cold/warm/moved coordinates, response spaces, packed replacement and
policy changes. Both Blackwell devices pass those nine cases under Compute
Sanitizer memcheck with zero errors. Host controls additionally cover tails,
row/column-major factors, insufficient scratch, BLAS failure, output/scratch
canaries and final-U lifetime. Full logs and trace files remain in ignored local
artifacts; bounded test summaries and original endpoint samples are retained here.

This does not establish an optimal schedule on every unmeasured GPU, all-functional
GPU4PySCF parity, or a combined gain with OpenBLAS/lazy range anchors. Those claims
require separate same-device integrated endpoint measurements.

## Replay and storage

```bash
PYTHONPATH=python:. python benchmarks/results/bounded-packed-projection-20261101/report.py
PYTHONPATH=python:. python -O benchmarks/results/bounded-packed-projection-20261101/report.py
```

The two payloads retain every original JSON byte. FP64 values use lossless XOR
byte planes keyed by method and JSON location; no decimal rounding or sample
selection is used. The reader reconstructs original formatting and SHA-256 before
checking identities, scheduler/device pairing, numerical gates, semantic work,
and complete cold/warm medians. Gates remain enabled under `python -O`.

To rerun, invoke `python -m benchmarks.readme_defaults standard --method <method>
--mode df --atoms 96 --warm-repeats 3 --force-tile-points 1024 --inputs
<qualified-inputs> --source-revision <full-base-revision> --output <ignored-path>`
in a finite GPU allocation. Alternate `GENERATIVEQC_DF_PACKED_PROJECTION` as
`generated/auto/auto/generated` after the declared conditioning owners. Preserve
scheduler visibility and record both the actual native identity and binary hash.

Two historical recovery manifests use authenticated lossless XZ storage to keep
these samples within the existing checkout cap. All original manifest bytes,
Git references and recovery paths remain available. The rationale is recorded in
[the projection decision](../../../.agents/notes/implemented/performance/2026-11-01-bounded-packed-projection.md).
