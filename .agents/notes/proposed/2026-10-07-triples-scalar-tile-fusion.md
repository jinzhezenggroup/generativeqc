# Proposal: fuse the occupied triples primal and W/V seed traversals

Status: proposed, opt-in only; native gates passed, large paired-force gate failed
Date: 2026-10-07
Related: #1763, merged #1999, #2042, #2045; deferred repeatability issue #2019

## Problem

The merged joint triples/Fock traversal reuses the six forward W cubes but
still executes separate energy, gathered W-seed, and six V-seed kernels for
each physical occupied triangle. Each W output invokes six individually
pruned scalar AD functions, recomputing compatible projections and repeatedly
reading the same virtual-coordinate source values.

## Proposed schedule

`occupied_triples_response.fused_tile_program()` composes the original primal
and its existing AD outputs. Each gathered W derivative rebinds inputs by its
inverse virtual-coordinate permutation. Ordinary TensorIR optimization/value
numbering shares equal producer nodes. There is no second CC equation set,
new amplitude iteration, precision change, or floating-point atomic reduction.

One grid-stride virtual-cube kernel publishes six W seeds, six V seeds, and
energy partials. The original block energy reduction tree, triangular energy
slot order, downstream packed reverse GEMMs, and complete Fock page domain
remain unchanged. Scientific derivative evaluations are not claimed to have
been omitted; only common scalar work is shared.

Keep six separate ordered denominator inputs. The algebraically symmetric
gap does not license using a differently ordered FP64 subtraction for a
permuted derivative. V boundary inputs similarly retain occupied and virtual
coordinate labels. W gathers preserve the existing virtual-label order.

## Admission and fallback

The joint owner admits six V cubes instead of one reusable cube: an additional
`5 * v^3 * sizeof(double)` before alignment. The W scratch remains six cubes.
If the optional allocation is not admitted, use the unchanged unfused schedule
before sacrificing existing page/panel residency. Exact minimal legacy-budget
refusal remains transactional. Full T3, full ovvv, and CPU mathematical
fallbacks remain prohibited.

The new flag defaults to false in both internal APIs. Gap-cotangent demand
also selects the retained unfused all-output schedule. Molecular force
composition still supplies mandatory full same-space Fock response. Standalone
primal, pullback, and full-Fock APIs remain qualification controls.

## Work and evidence contracts

The scalar region changes from eight launches to one per occupied triangle;
the downstream energy reductions, reverse GEMMs/audits, Fock kernels, Lambda,
orbital, and nuclear work are not credited as removed. Both paths write twelve
W/V seed values per virtual point; fusion increases simultaneously resident
seed storage rather than claiming to eliminate necessary derivative outputs.

Generated receipts count source scalar reads (including V products and ordered
denominator construction), seed/energy-partial writes, and FP64 scalar
arithmetic. They exclude finite checks, integer indexing, cache behavior,
compiler instruction transformations, and all parent phase work. They are not
measured memory traffic. The complete endpoint separately reports transfers,
admitted memory, phase timings, and scientific residual/stationarity checks.

The compiler graph has 376 prepared nodes instead of 1,743 summed nodes across
the separate scalar functions. Source-level modeled scalar reads per point
drop from 1,794 to 216, and arithmetic operations from 2,496 to 763. These
include ordered denominator and V product construction, not finite checks.
The compiler schedule identity binds the equation hash, coordinate and
denominator order, FP64 scalar emission, seed storage, and energy tree policy.
Node counts are not FLOP counts or a measured
speedup. CPU interpreted and compiled FP64 tests compare all thirteen outputs
to the original individual AD calls, including distinct ordered denominator
bindings. Native combined tests compare seven cotangents to the independent
host reverse composition, energy to original triples algebra, full Fock to its
standalone owner, transfer counts, page/panel fallback, and exact-budget refusal.

Local CPU qualification: 38 passed, 86 skipped (real-GPU tests require a finite
Slurm allocation). Compiler structure, CUDA ownership (333 files), vendor,
electronic-structure, provider-selection, and default-promotion inventories
were checked without errors. These CPU/structure results are not GPU gates.

On October 7, n2 Slurm job **2637** qualifies the generated CUDA owner:
**52 passed** across complete independent energy/force directions, original
triples energy/cotangents, full-Fock parity, compiler scalar checks, and the
fused/legacy exact-budget fallback tests. The validation adapter explicitly
requires selection of fusion for gap-free complete force calls. Memcheck runs
the two combined variants and admission/fallback case: **3 passed**, **0 errors**.
The allocation retains Slurm's `CUDA_VISIBLE_DEVICES=1` on an RTX PRO 6000
Blackwell (driver 595.91.07). Native library SHA256 is
`07273d961d77c1cd2f42812882174db1a03f670275c230016a1d5d56bcf98f4d`.
Whole-source verification and immutable receipts remain in
`qualification-2637/`; this is real-device evidence, not a modeled gate.

The first endpoint allocation (2638) stops at command-line validation before
molecular/GPU execution because the existing argument-count ceiling did not
include the new selector. The selector is appended at argument 22, its arity
is covered by 17 compiled-executable CLI tests, and `auto` at argument 21
retains the original RHF energy/density criteria rather than requiring a
tighter reference solely to address the new trailing option. This benchmark
fix does not change the already-qualified native scientific library. The
stopped sample is not an endpoint timing or numerical rejection.

Slurm job **2640** adds a 3-occupied/9-virtual native gate spanning multiple
energy CTAs and all 6/2/1 occupied degeneracies: **1 passed**. All seven
cotangents and full-Fock outputs agree with the retained unfused native owner;
energy independently agrees with the original virtual-triangle triples
inventory. The extra test source and library hashes are retained separately
under `multi-cta-2640/`, without changing the running endpoint source snapshot.

The first completed 28-AO ordered cold pair in job **2639** reports native
E+F wall time **44.404422247 -> 44.121034643 s**, with triples composition
**1.235489814 -> 1.069295430 s**. Total and triples energies are identical;
maximum force difference is **4.884981308350689e-14**. Maximum independently
reported Z residual is `2.8168603803345195e-13`, and stationarity is
`3.8276044660168207e-13`. The scalar-region launch ledger is **12,320 -> 1,540**
(1,540 triangles; 10,780 launches removed); seed workspace is
**28,672 -> 49,152 bytes**. This one ordered pair is not a statistically
qualified endpoint speedup and does not replace the 56/230-AO gates or a
complete Nsight launch/transfer census.

Offline `cuobjdump` inspection of the same native library reports **244
registers/thread, 3,072 shared bytes, zero stack and zero local bytes** for
the fused kernel. Register pressure is high despite the absence of spills;
modeled arithmetic reduction must not be interpreted as a hardware throughput
win. Retain the opt-in policy pending larger endpoint evidence rather than
assuming a CTA/register-limit-only experiment will establish acceptance.

Raw source/build/test/endpoint evidence stays ignored under
`.artifacts/1763-tile-fusion/` and on n2 under
`/data/jzzeng/triples-fusion1763-20261007/`. No Release or external backup is
authorized. The completed endpoint/census receipts below supplement the initial
qualification without authorizing a default promotion or a speedup claim.

### Completed endpoints: Slurm 2639

The allocation completed all six unprofiled endpoints, followed by both
28-AO Nsight profiles, with the same native library and benchmark executable.
Each unprofiled pair runs unfused then fused in separate cold processes, with
identical molecular input, tolerances, and selectors except argument 22.
Fusion is selected without resource fallback in every fused receipt. These are
single ordered pairs, not randomized repetitions or statistically qualified
speedups; instrumented timings are kept separate.

| Target | Native complete E+F seconds, unfused -> fused | Triples phase seconds | Maximum paired force difference | `atol=5e-10, rtol=0` |
| --- | --- | --- | --- | --- |
| 28 AO | 44.404422247 -> 44.121034643 | 1.235489814 -> 1.069295430 | 4.884981308350689e-14 | pass |
| 56 AO | 700.384432009 -> 698.856963992 | 21.646959567 -> 20.205735095 | 2.220446049250313e-13 | pass |
| 230 AO / 488 aux | 690.911690851 -> 666.611140023 | 39.266950292 -> 33.470825709 | **2.9812392554617873e-9** | **fail** |

Total and triples energies are identical in both small unprofiled pairs.
The large pair differs by `4.121147867408581e-13` in total energy and
`4.4009934585531596e-15` in triples energy. CCSD/Lambda/Z iteration counts
match: 21/23/14, 20/23/13, and 20/21/12, respectively. RHF counts match at
20 and 17 for the small pairs but **differ at 23 -> 20 for the large pair**.
Its reference phase changes from `151.037961903 -> 132.334370982 s`, explaining
most of the complete-time difference independently of the scalar fusion.
Do not attribute the entire large complete-time difference to this schedule.

Across each cold pair, the maxima of independent Lambda residual, Z residual,
and stationarity are, respectively:

| Target | Lambda residual | Z residual | Stationarity |
| --- | --- | --- | --- |
| 28 AO | 9.331867873006706e-13 | 2.8168603803345195e-13 | 3.8276044660168207e-13 |
| 56 AO | 3.744986304135408e-13 | 2.6641918072076755e-13 | 7.143886177063408e-13 |
| 230 AO | 6.115238039114635e-13 | 1.3650302921125122e-13 | 1.0210193801540868e-11 |

**The large paired-force gate is rejected.** The independent small/native
gates do not supersede it. #2019 remains explicitly deferred: neither its
deferral nor the earlier localization makes this new sample a pass or
establishes this sample's cause. No RHS/Z/nuclear localization, tolerance
relaxation, passing-rerun substitution, or default promotion is performed.

The profiled 28-AO pair returns identical total/triples energies and force
difference `5.417888360170764e-14`. Its instrumented complete timings are
`45.125718554 -> 45.035645174 s`, triples phase
`1.415220889 -> 1.368149822 s`; they are not clean timing repetitions. All six
pairwise comparisons among the four observed 28-AO cold/profile outputs pass
the unchanged force gate, with maximum `5.417888360170764e-14`. The ignored
analysis report retains every pair, both same-mode cold/profile comparisons,
all phase timings, residuals, and actual reference/solver iteration counts.

### Actual census versus scoped modeled work

The 28-AO Nsight SQLite records **413,130 -> 402,350 complete kernel launches**,
exactly the **10,780** launches removed from this scalar region. Within it,
the unfused energy/W/V kernels have 1,540/1,540/9,240 launches and summed device
durations `0.013525534 + 0.155676212 + 0.061133780 = 0.230335526 s`.
The fused kernel has 1,540 launches and summed device duration `0.054929441 s`.
The parent resolvent still has 4,200 launches in both profiles; reverse GEMMs,
Lambda, orbital/nuclear response, and other parent work are not credited as
eliminated. Summed device durations are not endpoint elapsed times.

Complete transfer counts and bytes are identical in both profiles:
H2D **288 / 19,200,015 bytes**, D2H **484 / 16,781,430 bytes**, and D2D
**478 / 53,368,448 bytes**. This is an actual complete 28-AO census, not a
claim that the larger endpoints have been fully profiled.

| Target | Scalar launches | Simultaneous seed workspace bytes | Modeled FP64 source reads | Modeled scalar arithmetic operations |
| --- | --- | --- | --- | --- |
| 28 AO | 12,320 -> 1,540 | 28,672 -> 49,152 | 1,414,533,120 -> 170,311,680 | 1,968,046,080 -> 601,610,240 |
| 56 AO | 91,840 -> 11,480 | 229,376 -> 393,216 | 84,357,611,520 -> 10,156,769,280 | 117,367,111,680 -> 35,877,847,040 |
| 230 AO | 1,320 -> 165 | 604,456,216 -> 1,036,210,656 | 3,195,090,794,610 -> 384,693,206,040 | 4,445,343,714,240 -> 1,358,893,130,595 |

Modeled scalar writes remain identical: **9,464,840**, **564,448,640**, and
**21,378,801,840 FP64 values**, respectively, including energy partials.
Multiply these logical FP64 read/write counts by eight for modeled bytes;
`analysis.json` retains both forms. They are source-level work receipts, not
measured DRAM transactions, executed hardware FLOPs, or eliminated parent
materializations. Simultaneous seed storage increases as already admitted;
there is no claim that fusion lowers this workspace.

### Device and memory interpretation

`TARGET_INFO_CUDA_DEVICE` maps trace logical device 0 to GPU inventory ID 3;
`TARGET_INFO_GPU` maps that ID to physical CUDA index 1 and UUID
`4b4be14f-ec84-6736-a7d8-968d62900c72`. This matches the Slurm-assigned
`CUDA_VISIBLE_DEVICES=1` and the NVML sampler's recorded UUID. No visibility
override is used. The 200-ms samples report total-device-used peaks:

| Target | Sampled total device used MiB, unfused -> fused | Reported numeric capacity bytes, unchanged |
| --- | --- | --- |
| 28 AO | 29,677 -> 29,677 | 215,324,092 |
| 56 AO | 29,679 -> 29,677 | 1,361,600,413 |
| 230 AO | 30,011 -> 30,011 | 7,107,919,129 |

These samples include device context/driver/other allocations and can miss
short-lived peaks; they are **not triples workspace or an isolated owner peak**.
The 28-AO profile's `memKind=2` allocation/free event high-water mark is
**136,700,650 bytes in both modes**. All 187 allocation/free pairs balance;
total recorded allocation volume increases exactly 20,480 bytes, from
312,475,153 to 312,495,633, matching the extra seed workspace. That event census
does not account for most of the sampled total-device-used figure.

Static inspection finds a **106,240-byte stack and 254 registers/thread** in
the generic `independent_jk_derivative_kernel`, which still runs twice per
28-AO profile (summed device durations `39.057690651 -> 39.021885870 s`).
Implicit driver stack/local-memory reservation is a plausible contributor to
the large device-used figure, not an established allocation attribution.
In particular, Nsight's `localMemoryTotal` is nonzero even for the statically
stack-free fused kernel; do not equate it with spill bytes. Exact isolated
owner/driver peak attribution remains unqualified, and no memory optimization
or causal force diagnosis is inferred from these receipts.

### Reproduction and retained artifacts

The endpoint executable SHA256 is
`9868556fee145f0386fac10cea330ba4f808a5ceb781b1f89157d3cf55eef331`;
the native library hash is the same one recorded for job 2637. Source/input/
binary manifests, JSON/time/NVML samples, and reproduction commands are retained
under `endpoints-2639/`; Nsight SQLite/`.nsys-rep` files stay on n2.
The compact all-pairs/work/census report is
`.artifacts/1763-tile-fusion/endpoints-2639/analysis.json`, with offline analysis
source `.artifacts/1763-tile-fusion/analyze_evidence.py`. Reproduce without
executing GPU work:

```bash
root=.artifacts/1763-tile-fusion
remote=/data/jzzeng/triples-fusion1763-20261007/endpoints-2639
ssh n2 "python3 - profiles $remote" < "$root/analyze_evidence.py" > "$root/endpoints-2639/profile-census.json"
python "$root/analyze_evidence.py" endpoints "$root/endpoints-2639" "$root/endpoints-2639/profile-census.json" > "$root/endpoints-2639/analysis.json"
```

Actual endpoint reproduction must continue to use finite Slurm `srun` on n2,
`main`, `--gres=gpu:pro6000:1`, retaining the assigned device visibility.
No new real-GPU run is needed to analyze these completed receipts.

## Rejected shortcuts and remaining gates

- Do not merge the six ordered denominators solely using mathematical symmetry.
- Do not count fewer scalar calls as fewer mathematical derivative outputs.
- Do not sacrifice a full occupied page just to fit optional fused seeds.
- Do not treat shared-source work from #2045 as newly removed by this schedule.
- Do not investigate or relabel #2019 as part of this scoped fusion work.
- Do not claim completion of general W/V/response fusion in #1763 from this slice.

Promotion requires independent energy/triples/full-force gates, unchanged
residual/stationarity criteria, finite-GPU sanitizer/resource qualification,
matched 28/56-AO and larger DF endpoint measurements, and complete launch/
materialization/transfer/peak receipts. Until then retain the unfused default.
