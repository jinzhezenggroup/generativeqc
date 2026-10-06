# Proposal: demand-aware and bounded parallel gap-response reduction

Status: proposed — explicit schedules are implemented and primitive gates pass;
the strict 230-AO cold-force gate fails, so defaults remain unpromoted.
Date: 2026-10-06

## Problem and source ownership

Issue #1763 identifies the generated scalar `gap_response_node_2` as an
underexposed reduction, not a Direct derivative spill/barrier tail. The
runtime-indexed emitter assigns the scalar to one thread and traverses the
entire virtual cube serially. Increasing only CTA size cannot fix this.

An additional complete-consumer inspection matters: `df_ccsdt_force.cu`
explicitly adds full `foo`/`fvv` Fock-resolvent sources and never adds the
epsilon cotangents from `pullback_df_cuda`. These sources replace the
diagonal epsilon derivatives, including internal same-space degeneracy.
Accelerating a discarded derivative alone is not the best endpoint schedule.

## Proposed implementation

1. Keep TensorIR/shared AD as the only scientific definition. A generic
   compiler region recognizes linear pointwise producers, independent
   reductions and linear output composition. The compiler-visible identity
   records the graph, fixed FP64 warp-tree policy, runtime tile bound and
   streamed producer storage. Nonlinear/nested or non-FP64 regions are rejected.
2. Stream the positive/negative gap-seed producers instead of materializing
   another cube. Share the scalar root across its three aliased outputs and
   compose the three virtual marginals in one kernel without intermediate
   vectors. Index maps come from reduction axes, not a second CC equation.
3. Split a scalar domain into at most 256 CTAs, targeting 4096 elements per
   tile, then drain bounded partials with a fixed FP64 tree. Vector outputs
   use one CTA per element. No floating-point atomics, FP32 or fast math are
   introduced. Inputs are immutable and do not alias the response arena.
4. Retain the original runtime-indexed source-major serial schedule and its
   capacity calculation for explicit oracle/compatibility comparisons.
   The current candidate keeps this as the default for canonical consumers;
   parallel execution is explicit and uses the ordinary provider stream.
5. Let the complete full-Fock consumer explicitly omit unrequested epsilon
   cotangents, their seed VJP, reduction, scatter, outputs and arena. Empty
   output vectors represent omission; zero vectors must not pretend to be
   computed derivatives. The compiler records the requested scalar-AD output
   identity and output count. The standalone API still requests all nine
   input cotangents by default.
6. Charge all nine primal inputs even when the last two derivatives are
   omitted. Complete admission separately charges borrowed inputs, detached
   requested outputs, GPU/provider storage and other simultaneously live
   caller state. Denominator/energy audits and full-Fock response remain
   mandatory. Resource refusal must not fall back to diagonal-only forces.

## Initial evidence, not endpoint acceptance

Source base: `c5ab37ec9e7b3d38d2e06729319f9eef66510e5c`.
The preceding #1972 exact J/K optimization is already merged; its action-count
or cache win is not credited to this work. Only n2 is used, with finite Slurm
`main` / `gpu:pro6000:1` steps preserving assigned device visibility.

Initial CPU compiler/independent triples suite: 65 passed, 77 GPU-only skipped.
GPU jobs 2444 and 2445 each pass 24 generic reduction tests against independent
`math.fsum`/NumPy signed-seed projections, including NaN/Inf refusal and aliased
outputs. Job 2445 uses runtime-bounded partial storage.

One 221-virtual seed receipt, device interval only:

| Quantity | Serial | Parallel/streamed |
|---|---:|---:|
| device seconds | 0.376993713 | 0.000337184 |
| launches | 8 | 3 |
| response workspace bytes | 86,356,200 | 3,824 |
| cumulative intermediate elements written | 21,588,606 | 256 |
| logical value reads | 64,764,050 | 43,175,700 |
| logical value writes | 21,588,828 | 478 |
| reduction summands including scalar drain | 43,175,444 | 43,175,700 |
| scalar error against `math.fsum` | 3.5083e-13 | 8.8818e-16 |
| maximum virtual-vector error | 7.9936e-15 | 8.6597e-15 |

These are single isolated adapter samples, not complete force timings or
statistical confidence intervals. Transfers/allocations are outside the
reported device interval. Counts are logical generated work, not hardware
DRAM transactions or total instructions; streaming recomputes producer
scaling and warp trees add bounded reduction overhead. The remaining seed/W
cubes are still part of the occupied-tile owner. No full T3 is materialized.

The first full library build exposed a toolchain mismatch: CMake selected the
Conda C++ compiler while adapter/executable links used system g++. Keep one
explicit system C/C++ toolchain and an explicit Conda Python interpreter;
do not repair this by suppressing ABI errors or weakening ccache identities.
The compiler/cache receipts and failed prototype logs are retained separately.

## Complete qualification receipts and retained negative evidence

All execution uses n2 only, finite Slurm `main` / `gpu:pro6000:1`, and the
assigned device visibility. The paired batches each retain one GPU throughout;
different batches received different physical GPUs. Driver: 595.91.07.
Compiler: explicit system GCC/G++ 11.4.0, CUDA 12.9.1, architecture 120;
ccache 4.5.1 and explicit CXX/CUDA launchers. A compiler change reset CMake's
cached architecture/interpreter options; a second explicit configure restored
them. Neither cache correctness nor the numerical gates were weakened.

Final validation source is separately exported to `qualification-source/`;
it never overwrites a source tree used by live endpoint jobs. Job 2461 reports:

- 45 occupied-response tests pass, including all nine native cotangents and
  demand omission at 221 virtuals, bounded panel replay/exact-budget refusal,
  unrequested epsilon sentinels, nonfinite epsilon inputs and corrected Lambda.
  Large validation inputs construct only required physical Gram blocks, not
  the tiny fixture's unnecessary full virtual-fourth-power ERI.
- Independent complete force tests pass 12 cases each with serial/all outputs,
  parallel/all outputs, omitted gap and the retained default: 48 passes. The
  Hamiltonian, finite differences, residual and stationarity gates are unchanged.
- Full generic/occupied-response memcheck passes 73 tests, with zero errors.
  Job 2450 separately passes all 28 generic tests under memcheck; job 2446
  passes those 28 without instrumentation, including scalar drain overflow.

Complete endpoint receipts use the same native library/executable and matched
selectors, including automatic exact J/K policy. The input geometries for
28/56 AOs are the original 12/24-atom reference-cluster rows, with retained
STO-3G shell metadata replicated per monomer. Their conventional reference
energies are **not** correlation-only DF energy oracles. All completed samples
are retained; energy/force differences use every measured serial/candidate pair.

| Target / sampling | Serial/all (s) | Parallel/all (s) | Omitted (s) | Max force difference, parallel / omitted |
|---|---:|---:|---:|---:|
| 28 AO / 28 auxiliary, median of 3 per mode | 44.668950 | 44.623063 | 44.612735 | 5.862e-14 / 4.796e-14 |
| 56 AO / 56 auxiliary, one cold sample per mode | 708.081384 | 706.565092 | 706.277965 | 1.457e-13 / 1.226e-13 |
| 230 AO / 488 auxiliary, one cold sample per mode | 829.693655 | 747.074795 | 734.211392 | **3.822e-9 / 4.468e-9 — fails** |

The recorded paired acceptance is energy difference <= 5e-11 and force
difference <= 5e-10, with Lambda/Z residual <= 1e-9 and stationarity <= 1e-8.
The 230-AO energy/residual/stationarity gates pass, but both force comparisons
fail. Do not exclude these samples, relax the threshold, claim accepted
complete-endpoint speedups, or promote either new default. The native force
owner and canonical API both retain serial/all-output defaults.

Before gap execution, the cold 230-AO primal denominator identities already
differ (`553148074470226251` versus `8193414523046746159`). Comparison with the
retained, previously merged serial endpoint also differs by 4.678e-9 in force.
This establishes unmatched floating-point primal payloads and comparable
legacy variability, **not** a proved root cause. Fixed-input 221-virtual
response comparisons pass without changing their existing tolerances. Further
matched-frame/upstream investigation is required; unrelated production
precision changes are not part of this patch. The strict large-factor
`atol=rtol=3e-10` gates remain untouched.

The 230-AO triples phase is 109.790965 / 44.188616 / 43.752140 seconds. Its
gap reduction counts are 1,320 / 495 / 0 launches and 86,356,200 / 3,824 / 0
workspace bytes; cumulative intermediate elements are 3,562,119,990 / 42,240 /
0. Full endpoint admission remains dominated by other owners:
7,107,920,953 / 7,107,920,953 / 7,107,919,113 bytes. Reference time also changes
between the cold samples; do not credit that unrelated variation to reduction.

Ordinary-dispatch Nsight Systems job 2450 captures a complete 58-AO endpoint
per mode. Actual gap-region launches, including seed/scatter, are 350 / 175 /
0. Parallel scalar/vector/drain launches each occur 35 times, with grids
37 / 53 / 1, 256 threads, registers 32 / 28 / 17 and static shared memory
64 / 192 / 64 bytes. The binary resource dump records zero stack/local storage.
Tracked complete-endpoint transfers are:

- H2D: 56,872,542 bytes in all modes; every primal input is retained.
- D2H: 43,577,959 / 43,577,959 / 43,577,495 bytes; omission removes exactly
  464 epsilon-output bytes and two copy operations.
- D2D: 123,376,120 bytes in all modes.

CUPTI tracked live allocation high-water is 152,805,843 bytes in all three
58-AO modes; it excludes untracked driver/context storage and pool reservation.
The 200-ms total-device memory samples peak at 29,677 / 29,679 / 29,889 MiB
for the 28/56/230-AO batches, unchanged across modes within each batch. These
total-device observations are not attributed exclusively to endpoint-owned
allocations and do not show a complete peak-VRAM win. Logical generated traffic,
numeric allowances, tracked live allocations and total-device residency are
distinct quantities.

Nsight Compute cannot access hardware counters (`ERR_NVGPUCTRPERM`). Its
failed receipt is retained; no privilege or driver-policy bypass is attempted.
Measured DRAM transactions are therefore unavailable. CUPTI launch/copy/resource
receipts and compiler-derived logical reads/writes must not be relabeled as
hardware DRAM measurements.

The small batch (job 2448) was deliberately stopped after the required 28-AO
repeats and first complete 56-AO triplet, before an unnecessary additional
56-AO repeat completed. Its cancellation and partial trace are retained; no
incomplete result is a numerical sample. Large job 2451 and qualification job
2461 finish normally. The earlier profiler job 2449 missed the explicit GPU
test flag and failed before instrumenting CUDA; it is not acceptance evidence.

Binary SHA256 used by these receipts:

```text
libgenerativeqc.so  0c6415da23eb797a4aa4547b2a44fe202798d05eaf8deb18053f89a68fd8b2e3
endpoint           8cb26c36240ba00d07c1968f268b2394de2faf4da7b04a94d2623d3fa4cc4dc2
```

Raw paths: `build-current/`, `unit-2446/`, `qualification-2461/`,
`endpoints-2448/`, `large-2451/`, `profile-2450/`, `endpoint-summary.json` and
`endpoint-gate-failures.json` under `/data/jzzeng/gap1763-20261006/`.

After the header's default-retention documentation and final validation-fixture
edits, the final library is rebuilt and requalified by job 2463: 45 native
response passes, 48 independent complete-force passes and 73 memcheck passes
with zero errors. Its SHA256 is
`b849a314102d9ece33483fbb5a5408ff02f3302e75b23bfc5c6fbd64438904a8`.
The executable/probe hashes remain unchanged, but the library hash does not;
the exploratory full-size timings above belong specifically to the earlier
recorded binary, not an accepted performance qualification of the final one.
Final focused CPU/compiler validation is 61 passes and 99 GPU-only skips.
Compiler, SCF, cross-method, vendor, default-promotion and 330-file CUDA
ownership checks pass, as do configured Python/C++ formatting checks.

## Required qualification before promotion/completion

### Cold-reference convergence diagnostic

The benchmark originally requested RHF energy/density tolerances of
`1e-12` / `1e-11`, while the independent complete-force probe requests
`1e-12` for both. The native reference owner honors tighter requested
criteria. This mismatch is a testable hypothesis for cold-reference
variability, not a demonstrated cause of the failed 230-AO force comparison.

A benchmark-only trailing argument twenty-one can now tighten both reference
criteria to a finite positive value at most `1e-12`; omitted controls retain
the original behavior. Original RHF energy change, density RMS and iteration
count are propagated as diagnostics without changing scientific equations or
replaying the reference. CC, Lambda, Z, stationarity, strict paired-force and
large-factor gates remain unchanged. Ten executable CLI tests validate the
legacy positional slots, tightening-only policy and rejection before GPU setup.

Job 2464 requests `1e-13` and fails at the first 28-AO RHF solve with the
configured 150-iteration limit. No CC/force endpoint completes, so it is
negative evidence, not an accepted sample. Do not describe this observation as
a proved numerical floor or increase solver limits to hide the failure.
Job 2466 instead requests the independent probe's `1e-12` criteria. Its
completed 28-AO serial/parallel/omitted samples and additional cold serial
repeat pass the original paired gates. Maximum force differences against both
serial references are `4.352e-14` (parallel) and `6.217e-14` (omitted); the two
serial runs differ by `4.885e-14`. Their final RHF density RMS is approximately
`1.679e-13`, below the requested criterion.

The first 230-AO serial solve instead fails RHF convergence under `1e-12`
criteria after 938.920 seconds in the native owner (939.19 seconds wall).
No large CC/force result is published and the remaining large schedules are
not attempted. Tightening alone therefore does not qualify the large cold
comparison or explain its previous discrepancy. Do not increase iterations,
relax the reference/response gates, or silently treat an uncompleted endpoint
as a successful comparison. The next diagnostic must hold the exact primal
payload and original molecular source/frame fixed across response schedules,
while explicitly charging retained copies and preserving complete ownership.
Original
legacy-tolerance failed receipts remain authoritative for their configuration
and are not superseded by a different reference tolerance.

The diagnostic build library SHA256 is
`d10838147c8af8957e5bf05eefb718a494b1b18735fa8d01b23c1f45824f9fdf`,
and endpoint SHA256 is
`c6e6b1aabc9ba3e37c9aff150f4db318ab8f1cda20ee81eec97cf4ee118c41c0`.
Its explicit ccache receipts report two hits and five misses across seven
cacheable calls. Full source verification agrees on all compiled/test inputs;
the historical remote Agent Note differs from the local note and is explicitly
recorded in the actual-source manifest and verification receipt, not silently
represented as the local version. Remote build and run directories are
`build-reference/`, `reference-pairs-2464/`, `reference-pairs-2466/` under the
same retained evidence root. The tightened-reference analyzer audits every
completed candidate against every completed serial reference and writes a
separate report; it never overwrites the original cold gate failures.
Uncompleted attempted endpoints are recorded as failures without fabricating
numerical samples. Focused local validation is 35 passes and 108 expected
GPU/executable-only skips; the initial missing-`PYTHONPATH` collection failure
is retained separately. Compiler, SCF, cross-method, vendor, default-promotion
and 330-file CUDA ownership checks pass, as do the configured formatters.

Diagnostic-build job 2468 completes normally: 45 native response passes,
48 independent complete-force passes across serial/parallel/omitted/default,
and 73 memcheck passes with zero errors. These receipts belong to the
`d1083814...` library, not the later combined source. During investigation,
remote PR head advances independently to `f04f92ebc80f37c39c1c986f8e83faf6d9a2774b`
with a generic rational-literal lowering correction and updated policy probes.
That commit is fast-forward integrated without overwriting its files. The
benchmark policy probe is extended for the appended reference control and
rejects argument count 23, preserving its earlier gap/default assertions.
The old argument-count expectation's local test failure is retained separately.
The combined compiler/native source requires a new rebuild and qualification;
the earlier binary's tests/timings must not be relabeled as its evidence.

The combined build is qualified separately by job 2472: 45 native response
passes, 48 independent complete-force passes across all four controls, and
82 memcheck passes with zero errors (the added rational-literal CPU cases
increase the prior 73-test census). Ten executable CLI tests pass on this
build; focused local compiler/triples/policy validation is 49 passes and
108 expected GPU/executable-only skips. All ownership, boundary, default and
configured formatting checks pass. Complete source verification matches all
8,214 manifest entries before qualification. Subsequent source changes only
append this evidence to the Agent Note; compiled inputs remain identical.

Combined library SHA256:
`44331c1ce3794c88a8c5e46f2d95f3644fbd6e14ca23f782e4f8ff85fc9a0e17`.
The endpoint/probe hashes are unchanged from the diagnostic build. Explicit
CXX/CUDA ccache launchers remain verified; the before/after receipts show
14 additional hits and two misses. Raw directories are
`build-reference-integrated/` and `reference-integrated-qualification-2472/`.
This qualification does not fabricate a completed large tightened-reference
endpoint, qualify the original failed cold gate, or promote a default.

### Remaining gates

- Actual complete native cotangents with parallel enabled, including a
  multi-CTA scalar case, and original independent energy/derivative gates.
- Omission tests prove all seven requested derivatives unchanged, epsilon
  outputs unrequested/unpublished, unchanged primal H2D/input charges,
  reduced detached-output charges and correct actual launch/work receipts.
- Independent complete CC energy/force finite differences, full same-space
  Fock closure, unchanged residual/stationarity and exact-budget refusal.
- Serial, fused-needed and demand-pruned complete 28/56-AO cluster and
  230-AO / 488-auxiliary comparisons. Keep all parent timings visible and
  independently recompute the final unscreened physical Z residual.
- Real memcheck, ordinary-dispatch Nsight census/resources and sampled actual
  device memory peak, distinguishing them from planned numeric capacity.
- Final compiler/ownership/boundary/style checks, source/input/native hashes,
  reproduction scripts, and current-state documentation. Promote the
  complete consumer's omitted-output default only after these gates pass.

## Revisit and boundaries

Do not claim completion of all general W/V/response fusion in #1763. Do not
remove epsilon derivatives from a fixed-canonical consumer that needs them,
or substitute diagonal sources for full-Fock response. A new output demand,
precision, topology or device family needs its own qualification. Do not
infer complete savings by subtracting a different allocation's profiled
kernel sum. Raw receipts remain ignored under `.artifacts/issue1763-gap/`
and on n2 under `/data/jzzeng/gap1763-20261006/`; no Release publication is
authorized or used.
