# Decision: automatic energy-only DF CCSD Q32 cap

Status: implemented
Date: 2026-10-10

## Problem

After the bounded one-sided ladder factorization in PR #2191, complete energy
endpoints still launch many short Q8 contractions. Increasing the existing Q
tile can amortize launch and packing overhead without changing the operator.
This is an endpoint policy change, not a new kernel or an FP32 precision path.

## Decision and scope

At the canonical `run_df_ccsdt_native` owner boundary, resolve a zero
`ccsd_batch_limit` to 32 for energy-only calls and eight for forces, before
passing a positive cap to the solver. Change that complete endpoint's omitted
default and the benchmark's omitted cap to zero. All positive overrides remain
explicit. Zero now means automatic endpoint selection, not a one-Q request;
use an explicit cap of one to request one-Q.

Do not change standalone `SolverOptions`, `run_rccsd_native_state`, diagnostic
probe defaults, Lambda/response settings or precision. In particular, the
energy observation does not justify enlarging the automatic force tile.
No CPU/PySCF/reference-oracle dependency enters production: the independent
oracle remains only a qualification consumer.

The solver already admits actual storage against its complete-owner budget,
dimension checks and allocation constraints. Retain smaller tiles, one-Q and
scalar fallbacks, provider allowances and all sticky physical failures. A cap
is not a promise that every device or shape obtains Q32. Preserve ascending
individual-Q FP64 accumulation and the independent original expanded replay.

## Rejected alternatives

- Unconditional two-sided dressing introduces catastrophic cancellation even
  for finite symmetric inputs accepted by the range certificate. See
  `../../rejected/2026-10-10-unconditional-two-sided-df-ladder.md`.
- Per-sweep Q-invariant packing reuse passes focused numerical and sanitizer
  checks but adds about 49 MB without a complete-endpoint win. See
  `../../rejected/2026-10-10-df-cc-q-invariant-packing.md`.
- A zero-amplitude shortcut does not help this physical endpoint, whose T2
  initializes from MP2. No such shortcut is implemented.
- Q16 is a useful bounded feasibility control, but the Q32 pilot is more
  promising. Do not run repeated campaigns on the rejected reuse route merely
  to turn a sub-percent CCSD-only observation into an endpoint claim.

## Qualification

The host policy test compiles the actual CUDA owner resolution statement with
the real public declaration and benchmark selectors; it does not maintain a
second selector implementation. Three tests pass on the integrated tree.
They cover energy/force omitted defaults, zero and positive caps
1/3/8/16/32, argument positions and existing explicit overrides.

Slurm 2823 rebuilds the actual automatic-policy CUDA library and benchmark
using ccache. All fourteen original generated CPU/CUDA artifacts are
byte-identical to the #2191 baseline. A real cap-zero energy endpoint selects
Q32 and passes independent total-energy 1e-8, (T) 1e-10 and original expanded
physical replay 1e-10 gates. The prior Q8/Q16 pair (2821) and explicit-Q32
pilot (2822) remain feasibility observations, excluded from ABBA medians.

Slurm 2824 runs four **fresh** complete energy-only endpoints in ABBA order
on node2/NVIDIA RTX PRO 6000 Blackwell, preserving scheduler device visibility.
Workload: ethane230/o9/v221/Q488, FP64, ordinary full DIIS8, 64 GiB correlation
budget, original expanded physical replay and ordinary FP64 (T). Baseline is
the qualified #2191 library with explicit Q8; candidate is rebuilt with real
cap zero resolving to Q32. The wall boundary includes fresh process startup,
native RHF, DF source, CCSD/admission/projection, independent replay, (T) and
teardown, not merely the iteration kernel.

| Median seconds | Q8 baseline | Automatic Q32 |
| --- | ---: | ---: |
| Complete process wall | 157.15332048316486 | 154.36847927700728 |
| Complete CCSD | 61.12902234800001 | 58.214733571500005 |
| CCSD iteration | 51.25447172757032 | 48.329268851066395 |
| Expanded replay | 9.664862454 | 9.674089103 |

Observed complete wall decreases 1.77% (1.0180402192x); complete CCSD decreases
4.77% (1.0500610172x). There are only two samples per arm: this is a scoped
observation, not statistical/global performance or force/Lambda/response
promotion. The replay is not claimed faster.

All four matched total-energy and (T) values are bit-identical across arms;
maximum total-energy difference is zero. All independent gates pass across
the four matched and four feasibility/default-integration samples. Errors are
about 2.20e-12 total, 8.45e-14 (T), 5.32e-13 replay r1 and 2.62e-13 replay r2.

## Executed work and memory tradeoff

Thirty-eight evaluations are admitted with zero pair refusals. Iterations,
evaluations, Q slices, setup, geometry/projection and semantic work match.

| Complete CCSD metric | Q8 | Q32 |
| --- | ---: | ---: |
| Q tiles | 2,806 | 1,096 |
| Accumulation calls | 3,294 | 1,584 |
| Q operations | 165,676 | 73,336 |
| GEMM calls | 54,862 | 25,792 |
| GEMM summands | 26,308,647,544,608 | identical |
| Contraction summands | 26,799,066,988,432 | identical |
| Logical packing bytes | 3,270,860,758,368 | 2,934,115,738,848 |
| Logical accumulation bytes | 2,990,275,612,480 | 2,555,927,021,440 |
| Numeric capacity bytes | 4,576,754,470 | 8,945,140,974 |

This removes 1,710 primary tiles/accumulations, 92,340 Q operations and 29,070
GEMM calls, **not arithmetic summands**. Logical packing and accumulation
traffic decrease 336,745,019,520 and 434,348,591,040 bytes. These counters are
not measured DRAM transactions. Numeric capacity rises 4,368,386,504 bytes
(about 4.07 GiB); it is neither RSS nor complete peak physical memory. Do not
describe this policy as memory-free or infer work reduction from memory size.

## Source and retention boundaries

Both arms use the same frozen native RHF/source, reconstructed from parent
`d4b44b06489d44a517d3dfaecc6b7269f5b89b7a` plus the pinned copy-elision and
one-sided dressing patches. The candidate adds only the five-file endpoint
policy patch. Observed master `cd0eb059fd1d0ba205c7db160e50c1145cb8488e` before
measurement contains later opt-in SCF ordered Gram caching, absent in both
measured archives. Integration observes
`29d901a22bed9507ab3bf522a2fbd3692ab1ca40`, adding allocation-journal tooling.
Neither changes the five candidate files or the relevant CC implementation.
The pre-PR check also observes
`82c166caca181fc0df51b1ff9378b972e3b1eba8` (opt-in GFN2 density receipts),
with no drift in those candidate or CC files; no unrelated campaign is rerun.
Do not relabel these receipts latest-master recomposition, credit unrelated
SCF speedups, or rerun unrelated tests solely because master advanced.

Retain all eight endpoint records, exact recipes, provenance, numerical gates,
artifact hashes and minimal measured-source reconstruction in
`benchmarks/results/df-cc-energy-q32-default-20261010/`. The aggregate evidence
cap remains 67,108,864 bytes. To fit, only this loop's immediately preceding
eight-file #2191 bundle moves to existing merged Git history, pinned to
`f81640c8cd7531be60c1d9a4d2323e90934e1d1c` with every original byte count and
SHA-256 in its sibling `snapshot.manifest.json`. The existing restore tool
verifies all eight originals before writing. No accepted samples are lost,
unrelated evidence deleted, cap raised or external archive/Release published.
The prior rationale remains historical, with an appended recovery pointer.

## Successor evidence retention

The bounded-workspace successor migrates the six original files losslessly to
existing merged commit `9f67e7806e3151454530baf0ee66ae8808d826f0` (#2197).
The sibling `snapshot.manifest.json` pins all 19,127 bytes, and the shared offline
restore tool verifies every original file before materializing it. This keeps
all accepted samples and the fixed checkout budget. See
`2026-10-10-df-cc-explicit-blas-workspace.md` for the successor decision.

## Revisit when

A smaller-memory device, occupied-heavy shape or different bottleneck justifies
another bounded cap; or fresh independent force/response evidence supports a
different default for those endpoints. Keep explicit overrides and the
complete resource, independent replay and endpoint-work gates. Treat larger
tiles as a measured memory/performance tradeoff, not a universal heuristic.

## References

- `2026-10-10-df-cc-ladder-dressing-factorization.md`.
- `2026-10-10-df-cc-native-copy-roundtrips.md`.
- `docs/developer/df_ccsdt.md`, current endpoint policy.
- Ignored local `.artifacts/df-cc-energy-q32-default-20261010/` and node2
  `/data/jzzeng/qc-cc-energy-q32-default-20261010/` for transient build logs.
