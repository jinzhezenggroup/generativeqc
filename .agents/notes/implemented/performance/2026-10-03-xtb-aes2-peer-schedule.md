# Decision: bounded AES2 peer evaluation with ordered accumulation

Status: implemented
Date: 2026-10-03

## Problem

The fresh-SCC molecular endpoint still trailed xTBloom at 96/192 atoms after
runtime retention and identical-spin occupation sharing. A diagnostic profile
that exposed the existing bounded SCC fallback showed approximately 0.448 ms
per 96-atom AES2 potential invocation and 1.665 ms per terminal AES2 VJP.
The ordinary device-launched SCC graph profile omits internal kernels, so its
visible kernel list must not be treated as complete work accounting. The
diagnostic fallback binary was isolated and never used for acceptance timings.

## Decision

`method/gfn2_aes2_schedule.py` selects the fused existing route for rounded-up
mean atom counts at most 32. Larger batches use at most 256 atom tiles per
system, with 32 lanes per block. Each block strides over actual ragged target
atoms and evaluates a chunk of independent peers into fixed shared storage.
Ten potential component owners preserve their respective peer accumulation
orders. The VJP owner consumes four components in the original peer/component
order and chooses the first failing peer before recording a system failure.

The runtime validates metadata, multipoles and pair caches once per system,
then launches the evaluation tiles. A separate publication launch suppresses
every output of a failed system. Existing scratch buffers, SCC activity gates,
Graph capture and the fused route remain available. No allocation, D2H copy,
host synchronization or previous SCC solution is added.

Generated TensorIR/AD mathematics, pair indexing and FP64 compilation remain
unchanged. Materializing a pair VJP changes NVCC's FMA contraction boundaries:
the native oracle observed one-ULP differences, while an isolated diagnostic
build with `--fmad=false` was bitwise equal. Production flags retain FMA. Tests
require bitwise potential equality and VJP differences at most
`8 * epsilon * max(0.125, abs(reference))`, with exact failed-output sentinels.
Independent tblite, centered force differences and xTBloom endpoint gates
remain necessary in addition to this schedule comparison.

## Work and resource contract

For a healthy system with N atoms, either route performs N onsite potential
evaluations and N(N-1) ordered pair potential evaluations. The VJP likewise
performs N(N-1) pair evaluations, including its existing geometry/cache checks.
The triangular cache contains N(N-1)/2 entries. Validation scans each input and
cache entry once per invocation, not once per tile. Chunking may evaluate later
peers speculatively after an earlier peer has failed, but publication and error
selection remain guarded. Healthy-system work does not increase.

The large route adds one evaluation launch, reuses global unpublished scratch,
and needs less than 3 KiB shared storage per block. Tile counts and peer storage
are bounded independently of molecule size. Mean-based selection needs no host
copy of the ragged offsets; highly imbalanced batches can deliberately select
the slower fused route, which remains correct.

## Rejected alternatives and synchronization lessons

- Distributing target atoms without parallel peer evaluation barely improved
  the endpoint. One lane still serialized all generated pair work.
- Revalidating inputs in every atom tile would amplify cache scans. Validation
  must remain one launch per system.
- A 128-tile cap left some 192-atom blocks processing a second atom. Raising
  the bounded cap to 256 reduced the measured large endpoint further.
- Parallel tree reductions would alter peer summation order. Independent
  component sums retain that order without serializing all ten components.
- A shared admission flag cannot be immediately reused after a conditional
  read. Its readers need a closing barrier before the onsite writer starts.
- The VJP owner's evolving failure code must be local until chunk completion.
  NVCC predication otherwise allowed inactive warp lanes to read the shared
  status while lane zero rewrote it; racecheck exposed this despite passing
  numerical tests. Only the completed chunk status is shared between barriers.

## Evidence

Qualification used finite Slurm allocations: n1 with `gpu:5090:1` for native
memcheck/racecheck, and n2 with `gpu:pro6000:1` for complete endpoint timing.
The native harness selects the fused oracle by padding the same original
systems with independent single atoms. It covers 31/32/33, 127/128/129 and
255/256/257 boundaries, 3/193/17 and 1/513/2 ragged batches, grid strides,
ordinary/SCC potential and VJP, two Graph replays, late NaN/cache errors,
finite overflow, stale geometry cache, invalid seeds, inactive systems,
closed SCC sequences, whole-system suppression and healthy peer independence.
Final memcheck: zero errors. Final racecheck: zero errors and warnings.

n2 endpoint qualification passed all 41 public CPU/CUDA, lifecycle, tblite and
force tests. Ten cases each include one cold, five repeated and five changed
geometry calls, all fresh SCC, FP64, 300 K, Broyden history 8/mixing 0.4 and
unchanged convergence tolerances. All 110 samples matched SCC iteration counts.
Maximum differences from xTBloom were 5.69e-14 Eh and 5.15e-15 Eh/bohr,
within the comparator's 5e-7 energy/force gates. Relative to the previous PR
binary, energy differences were zero and force differences at most 4.17e-17.

Representative n2 complete endpoint medians in milliseconds:

| Case/mode | Prior spin-sharing PR | AES2 schedule | xTBloom |
| --- | ---: | ---: | ---: |
| 24 atoms, repeated | 37.406 | 37.416 | 41.680 |
| 96 atoms, repeated | 132.364 | 125.088 | 129.660 |
| 192 atoms, repeated | 331.753 | 311.529 | 314.453 |
| 96 atoms, changed | 132.380 | 125.100 | 129.661 |
| 192 atoms, changed | 331.855 | 311.753 | 314.753 |
| 96 atoms, cold | 252.647 | 242.082 | 241.389 |
| 192 atoms, cold | 423.508 | 398.121 | 386.680 |

All ten repeated/changed medians beat xTBloom in this run; cold cases still
include regressions versus xTBloom. The approximately 1% 192-atom advantage is
small and does not establish universal hardware/size superiority.

A separate n1 RTX 5090 allocation confirmed all 41 public tests and all 110
sample gates. Repeated 96/192-atom medians were 124.682/312.304 ms versus
xTBloom 128.709/314.406 ms; changed-geometry medians were 124.575/312.117 ms
versus 128.727/314.095 ms. All ten repeated/changed medians again won, with
only a 0.6–0.7% margin at 192 atoms. The 192-atom cold call remained slower:
396.748 ms versus 389.489 ms. Receipts: `.artifacts/n1/aes2-cap256/`.
Here “cold” means the first call of a new calculator; only the first case also
pays process-level initialization. This is not an independent process launch
for every cold sample.

Source baseline: PR #1724, scientific commit `2d6bd50dc` (subsequent CI fix
only adjusts an AST test after the runtime wrapper move). xTBloom source:
`2cbdf1db8661ccbd5cb7d3d4bfc868a848cbbff3`. CUDA 12.9.1, Release sm_120,
one BLAS thread, scipy-openblas32 0.3.34.0.0. Binary SHA256 values:

- Prior: `04fbf8d16ebdd94f991313da723309ad45eea058e375fcb1b69104781ccc24d8`.
- Candidate: `213babf344a7bf90eaa194554dd9a3f98276e1c99f10e2954992f18b27cf9256`.
- xTBloom: `6be47a7183a21c10c8204b6fbeb27e8587bc45d1958e59f5c7d61356777db53f`.

Ignored local receipts: `.artifacts/n2/aes2-cap256/`,
`.artifacts/aes2-qualification/n1-cap256-sanitizers.log`. Reproduce with
`benchmarks/compare_xtbloom.py --waters 8 32 64`, then compare all engine JSONs.
Profiler timings above are diagnostic evidence, not endpoint acceptance.

## Revisit when

Reconsider thresholds for a measured batch/device mix, or use a generated pair
evaluation/materialization strategy that reduces work while preserving the
same failure and scientific gates. Cold setup, eigensolver, mixing and other
force consumers remain separate optimization opportunities. Do not drop the
bounded fused route or weaken gates to extrapolate these measured wins.
