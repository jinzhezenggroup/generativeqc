# Decision: bound and distribute GFN2 electronic matrix work

Status: implemented
Date: 2026-10-03

## Problem

The generated S/D/Q improvement in [the preceding note](2026-10-02-xtb-shell-common-work.md)
left complete GFN2 endpoints behind xTBloom. Native CUDA Hamiltonian assembly
used one block per system; density used one block per system/channel. Although
outputs are independent, one block traversed an entire matrix. Density also
performs a complete orbital reduction for each pair, magnifying this bottleneck
inside every SCC iteration.

## Decision

The compiler's `method/gfn2_electronic_schedule.py` owns the 256-thread block
width and a maximum of 128 matrix tiles per system. It emits host policy into
the existing electronic CUDA artifact, without importing the runtime or probing
hardware. Native wrappers consume that policy for Hamiltonian assembly and
restricted/spin density contraction.

The host descriptor exposes total matrix elements and batch size. The ceiling
of the mean extent selects the number of tiles; each system traverses its true
device extent using a grid stride. The cap bounds launch amplification for
ragged systems. Small matrices retain one block. Imbalanced batches may leave
some blocks idle or require extra passes, but need no metadata download,
allocation, global cache, new scratch or unbounded specialization.

One triangular pair owns both symmetric entries. Density retains its complete
ordered orbital sum per lane, including intermediate finiteness checks. Trace
reduction geometry stays at 256 threads and is not tiled. Validation, error
canonicalization, and whole-system publication retain separate stream-ordered
launches. An error in either spin channel or any tile prevents publication of
all outputs for that physical system. ABI layouts and SCC controls are unchanged.

## Semantic work

For 192 AOs (water-32), both versions assemble the same 18,528 triangular pairs.
Density performs 3,557,376 orbital contributions for each of P and W per
channel per SCC iteration; the compiler changes ownership, not these counts.
The old density block requires up to 73 matrix-pair passes per lane, while the
128-tile schedule needs at most one. Hamiltonian's full-square traversal changes
from 144 passes per lane to at most two, still evaluating only one triangle.
There are no added kernel launches or scratch bytes. Tiling does not bypass the
runtime's setup smoke calculation, retained-state rules or fresh SCC work.

## Validation

`tests/native/test_gfn2_electronic_schedule.cu` calls the actual native CUDA
entry points on imbalanced 3/7/17, 3/193/17, and 3/389/17 AO batches. It covers
one tile, multiple tiles, the cap with further grid-stride passes, restricted
and mixed spin layouts, CUDA Graph capture/replay, inactive members, NaN in a
late Hamiltonian tile, and density overflow from finite coefficients in a late
pair of the second spin channel. Independent long-double formulas check matrix
entries and density diagnostics; failed systems retain output sentinels.
Memcheck and racecheck pass with zero errors/hazards on n2's PRO 6000; the initial
Hamiltonian-only harness also passed memcheck on n5's RTX 5090. All execution
used the corresponding finite Slurm `main` GPU allocation.

The compiler/provenance suite passes 14 tests (two explicit CUDA opt-ins skipped
on the host). The complete CPU/CUDA endpoint/tblite suite passes 23 tests on n2.
Compiler structure, CUDA ownership, and all pre-commit checks pass.

## Complete endpoint evidence

The final cohort ran alone on an n2 PRO 6000 Blackwell Slurm allocation (job
2039), Release sm_120/CUDA 12.9.1, with the same LP64 scipy-openblas32 provider
and one BLAS thread. The baseline is PR #1714's previous head `6cf534bc7` based
on `cdd9e56cb`; the candidate is based on `06459d469`. The intervening master
change bounds non-xTB range-Hermite workspaces and does not change this method.
xTBloom is main `2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3`.

Every sample starts fresh SCC: 300 K, modified Broyden history 8/damping 0.4,
300 maximum iterations, energy tolerance 1e-10 and charge tolerance 1e-8.
The public energy/force endpoint includes host-visible preparation, transfers,
SCC, forces and publication. xTBloom additionally returns atomic charges.
Seven independent tblite fixtures plus 24/96/192-atom water clusters each retain
one cold, five warm and five non-rigid changed-geometry calls.

| Case/mode | Previous PR (ms) | Tiled (ms) | xTBloom (ms) | Gain vs previous PR |
| --- | ---: | ---: | ---: | ---: |
| Water-8 warm | 55.295 | 54.227 | 41.663 | 1.02x |
| Water-32 cold | 345.324 | 262.299 | 240.619 | 1.32x |
| Water-32 warm | 246.949 | 176.872 | 129.557 | 1.40x |
| Water-32 changed | 246.950 | 170.964 | 129.635 | 1.44x |
| Water-64 cold | 1078.339 | 450.244 | 384.872 | 2.40x |
| Water-64 warm | 1070.657 | 441.871 | 314.466 | 2.42x |
| Water-64 changed | 1071.351 | 442.195 | 314.811 | 2.42x |

Small-molecule warm ratios versus the previous PR are 0.988–0.999, with no
claimed benefit where one tile is retained. The small absolute differences are
at the noise scale; these samples do not establish superiority on tiny systems.
The complete endpoint **still trails xTBloom**, so the overall goal is unfinished.

All 110 candidate/reference sample pairs pass the 5e-7 energy and force gates,
without filtering on iteration count. Energy is identical to the previous PR;
maximum force difference is 4.17e-17 Eh/bohr. Against xTBloom, maximum errors are
5.69e-14 Eh and 5.17e-15 Eh/bohr. Every SCC iteration count agrees.

Measured binary SHA-256:

```text
previous PR: e974be8ab32d999b7cea855cd2ba36a792842247eac28fdc42201adae52c17af
tiled: eaa57fab4974101b8ce3936c61ed762f3a7bc90908d80d584ad4bd0c32cbb2e8
xtbloom: 6be47a7183a21c10c8204b6fbeb27e8587bc45d1958e59f5c7d61356777db53f
```

Raw reports, including all scalar/vector outputs and iteration counts, remain
ignored under `.artifacts/n2/final-tiles/` in the qualification worktree and
`n2:/home/jzzeng/xtb-perf-20261003/qc/.artifacts/final-tiles/`. Reproduce with
`benchmarks/compare_xtbloom.py --engine generativeqc|xtbloom --waters 8 32 64
--output <scratch.json>` inside a finite Slurm allocation, then compare using
`--reference`, `--candidate`, and `--output`. Use `gpu:pro6000:1` on n2 and
`gpu:5090:1` on the 5090 nodes, preserving Slurm's device visibility.

## Rejected alternatives

- Hamiltonian tiling alone reduced water-32 from about 245 to 240 ms on n2,
  insufficient to explain or remove the endpoint gap. Density must also expose
  independent outputs to multiple blocks.
- Splitting density's orbital reduction or replacing it with unordered atomic
  sums is unnecessary here and would alter reproducibility/finite-range gates.
- A Gaussian raising/lowering derivative prototype shared existing overlap IR
  but increased emitted scalar temporary counts: s/s SDQ 171 to 215, d/d summed
  Cartesian branches 11,202 to 13,875, and d/d shared CPU block 3,906 to 4,467.
  No production change was made from that prototype. Revisit only with a more
  effective factorization/schedule and independent derivative gates.
- n4 was unavailable for qualification: its loaded driver kernel was 580.173.02
  while its NVML library was 580.178. No system driver files were changed.

## Revisit when

An explicit device-resident ragged work list or a target-qualified schedule can
reduce empty tiles without metadata staging, or a separately qualified density
lowering can improve memory reuse while preserving numerical acceptance. The
complete public endpoint still needs setup/lifetime work: public singlepoint
currently rebuilds its runtime and repeats setup science. Do not infer xTBloom
superiority from a faster isolated contraction.
