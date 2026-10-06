# Proposal: demand-aware and bounded parallel gap-response reduction

Status: proposed — explicit schedules are implemented and primitive gates pass;
the strict 230-AO cold-force gate and latest same-primal omission gate fail,
so defaults remain unpromoted.
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

### Same-primal diagnostic evidence

The shared-primal diagnostic is implemented without a replacement scientific
map. Its library SHA256 is
`39bca1ae5df8a78fdc66c69732e72498b5e07830e234653197fba0819cb34ec5`;
the comparison executable is
`cbf2bc2555d634551ce7e0684b94b29eca2b96b096df3c751b96c3e7f9fa8b43`,
and the complete-force probe is
`8c20dd30a39ac9607d4991d614c56cd0834364120f2681e005a95b036abfc971`.
All 8,216 source-manifest entries match before qualification. Explicit CXX/CUDA
ccache launchers remain verified; build receipts record two additional hits
and six misses without resetting the shared cache.

Job 2474 passes 15 force tests, including three new diagnostic cases, and three
new memcheck cases with zero errors. The successful diagnostic fixture inserts
a physical oxygen p shell before the hydrogen shells, so the parallel path has
at least four virtual orbitals. Each response is compared with both serial
references at the unchanged energy `5e-11`, force `5e-10`, Lambda/Z `1e-9` and
stationarity `1e-8` gates. Independent PySCF energy/triples and two-step force
finite differences pass. Budget-zero/one refusal leaves all output sentinels
untouched. Initial job 2473's shell-order oracle rejection is retained, not
relabeled as a numerical result. Focused local validation is 49 passes and
111 expected skips; compiler/SCF/vendor/default/boundary, 330-file CUDA ownership
and configured formatting checks pass.

Job 2477 completes the 230-AO / 488-auxiliary diagnostic using exactly one
native cold primal. All cases retain denominator identity
`6315232735855893916` and nine-input bit-pattern census
`2674132729717729229`. Energy differences are zero, and complete forces pass
against **both** serial references:

| Composition | Response-only seconds | Maximum force difference |
|---|---:|---:|
| Serial/all | 507.087754 | 1.962383589e-10 |
| Parallel/all | 441.797895 | 2.338831351e-10 |
| Omitted | 441.374586 | 2.899804841e-10 |
| Serial/all repeat | 507.570574 | 1.962383589e-10 |

The serial repeat establishes nonzero response variability even with unchanged
native inputs; these are tolerance passes, not a claim of bitwise forces. Common
primal work takes 286.577691 seconds, cloning takes about 0.22 seconds per case,
and total comparison time including original-owner retirement is 2185.512606
seconds. These response timings are **not cold endpoint speedups**. Corrected
Lambda and full-Fock/source semantic work are unchanged; Lambda residual is
`6.116e-13`, the largest Z residual is `1.367e-13`, and the largest stationarity
residual is `1.492e-11`.

The retained original host state is 358,801,448 bytes, shared DF source is
49,259,323 bytes, fixed force outputs are 768 bytes, and clone admission is
769,102,267 bytes. Planned complete numeric capacity is 7,466,723,169 bytes for
serial/parallel and 7,466,721,329 bytes for omitted, within the 64-GiB budget.
The whole-diagnostic 200-ms samples observe a 29,977-MiB device peak; this is
not interchangeable with planned numeric capacity or a per-case device peak.
Gap launch/work telemetry remains explicit: 1,320 / 495 / 0 region launches
and 86,356,200 / 3,824 / 0 workspace bytes for serial/parallel/omitted.

Native RHF diagnostics are forwarded verbatim: 21 iterations, energy change
`1.0800249583553523e-12` and density RMS `9.756517294903296e-12` at requested
energy/density tolerances `1e-12` / `1e-11`. The existing shared HF stopping
policy includes its energy-scaled FP64 roundoff guard; this change does not
alter it or claim the stronger unguarded reference criterion. The analyzer
explicitly retains `cold_gate_superseded: false`. Shared-primal success narrows
the investigation but does not prove the original cold discrepancy's cause.

Receipts are `build-same-primal/`, `same-primal-unit-2473/`,
`same-primal-unit-2474/` and `same-primal-large-2477/` under the retained n2
root. Full combined qualification job 2478 passes 45 native response tests,
48 independent complete-force tests across serial/parallel/omitted/default,
and 82 memcheck tests with zero errors, on this exact library. Its receipts are
`same-primal-qualification-2478/`; the new diagnostic's three memcheck cases
remain separately recorded in job 2474, not double-counted in that 82-test run.
A separate, ignored exploratory
benchmark wrapper changes only the existing RHF DIIS history to four, keeps
both requested RHF tolerances at `1e-12`, and reuses the frozen force owner.
Its executable SHA256 is
`484b7de849c545bd9cd3824ceb67fea53688d1ba8d634ebf91c6152b71bf13f3`;
ten compiled CLI rejection/acceptance tests pass. Job 2483 starts only after
qualification success and explicitly checks the shared-primal result and
frozen library before proceeding; its outcome is recorded separately below.
This is a hypothesis test, not a production
policy change or a new acceptance gate. No running candidate source/library
is replaced, and only n2 is used.

### DIIS-four cold experiment and upstream integration

Job 2483 completes all four 28-AO cold cases at requested RHF energy/density
tolerances `1e-12` / `1e-12` and reference DIIS history four. Against both
serial references, maximum force differences are `3.598e-14` for serial,
`4.175e-14` for parallel and `4.441e-14` for omitted; energy differences are
zero. All reference solves take 26 iterations, with reported energy change
zero and density RMS `1.642e-13`. These receipts still belong to the frozen
`39bca1ae...` library and the ignored wrapper, not a production default change.

The first 230-AO serial attempt again fails RHF convergence after 938.95 seconds
at the fixed 150-iteration limit. No large CC/force result is published and
the remaining large compositions are not attempted. Thus DIIS four does not
repair the tightened large-reference convergence gate. The complete failed
attempt, sampled memory and `failed:1` status are retained in
`reference-diis4-2483/`; its report keeps
`legacy_cold_gate_failures_superseded: false`. No acceptance criterion is
weakened and this is not proof of the cold discrepancy's root cause.

Commit `c194df14de921d3b52bc656e08a8e81ef92e8dac` publishes the diagnostic and
the preceding qualification evidence. Its complete source patch is retained
as `same-primal-qualified-source.patch`, with a machine-readable receipt.
The earlier working-tree patch is retained too, but excludes files that were
then untracked; it is not a complete reproduction patch. All 8,216 compiled
manifest entries match this commit except the subsequently appended historical
note. The initial/qualified manifests differ only in the Python shell-order
fix, not in compiled scientific inputs.

Upstream advances to `88589bdf46b04317cacba505cde5e39b60423286`, including
public DF-force diagnostic publication. The only merge conflict is between
adjacent anonymous-namespace helpers; both primal census/admission helpers
and `publish_force_diagnostic` are retained, including the completed owner's
publication call. This does not alter response equations or gap defaults.
Imported historical experiment-patch whitespace is left unchanged; the
gap PR's own diff passes whitespace checks.

The combined source tree before this evidence append is
`f754f81c686caf990a8670d41aafd4b0e5c77edb`. All 8,262 manifest entries match
both before and after an isolated cached rebuild, without replacing the
running experiment's checkout or library. The new library SHA256 is
`e2d12f289c12b6e094b729bd502cff6e4c77f77db1d39a7872d67f19edd2ac64`,
and the same-primal executable is
`8805ab3c6ba98c9f1710d36482b119f947f8b83c672d075bcc1382c3729e56e3`.
The build records 455 cache hits and 12 misses with verified explicit CXX/CUDA
launchers and checkout-root normalization. Local focused/public-publication
validation is 73 passes and 112 expected skips; compiler, SCF, vendor,
default-promotion, cross-method, 330-file CUDA ownership, generated method
metadata and configured formatting checks pass.

This newer binary requires its own qualification; the earlier large diagnostic
must not be relabeled as its result. Jobs 2484, 2485 and 2486 respectively run
full native qualification, same-primal/public-force tests and memcheck, and the
230-AO shared-primal diagnostic. They are ordered by Slurm dependencies, use
only n2, and have explicit finite limits. Their results are pending at this
checkpoint. The original failed cold gate and unpromoted defaults remain.

Jobs 2484 and 2485 subsequently complete on the `e2d12f28...` library:
45 native response tests, 48 independent complete-force tests across all four
controls, and 82 memcheck tests with zero errors; separately, 15 force-owner
tests, one public H2 force/energy finite-difference test with real published
response diagnostics, and three new diagnostic memcheck cases with zero errors.
The upstream public publication path is therefore checked, not dropped to
resolve the helper conflict. Job 2486 is running on the isolated combined
source; its large comparison is still pending. No live candidate source or
library is overlaid when adding these historical receipts.

### Integrated large same-primal rejection

Job 2486 returns all four complete responses on the `e2d12f28...` library,
then exits one because the unchanged force acceptance check rejects omitted
outputs. The completed JSON, trace, timings, allocation, source manifest/patch,
memory samples and gate report are retained in `main-integrated-same-large-2486/`.
`endpoint-status.txt` is `endpoint-complete`; there is no successful overall
status. This is a completed numerical rejection, not an interrupted solve.

| Composition | Response-only seconds | Maximum force difference against both serial references |
|---|---:|---:|
| Serial/all | 509.569973 | 4.149791621e-10 |
| Parallel/all | 443.295707 | 3.887175026e-10 |
| Omitted | 442.881308 | **5.520250923e-10 — fails** |
| Serial/all repeat | 509.462872 | 4.149791621e-10 |

Omitted differs from the initial serial by `5.520250923041203e-10`, from
parallel by `5.25763432790427e-10`, and from serial-repeat by
`1.3704593015972932e-10`. The serial-repeat difference is itself
`4.14979162144391e-10`. All energy differences are zero; Lambda residual is
`6.116e-13`, maximum Z residual is `1.365e-13`, and maximum stationarity is
`2.514e-11`. All physical-source, full-Fock and Lambda semantic-work counts
are unchanged, and numeric admission remains within the 64-GiB budget.
The whole-diagnostic device peak remains 29,977 MiB over 10,996 samples.

Within this run the original source/frame, denominator identity
`16930483910286437394`, and nine-input bit-pattern census
`5974276677275782043` are shared and guarded throughout. Common primal time is
294.425535 seconds and total comparison time is 2200.735155 seconds. These
response-only timings still do not qualify independent cold performance.
The report explicitly retains `cold_gate_superseded: false`.

Thus cold-reference variability alone cannot explain every observed large
rejection. Nor do these results prove that omitted cotangents caused the
failure: the serial-repeat variability is material, and no intermediate
seven-cotangent/Lambda/physical-source bit-pattern comparison has yet been
recorded. The next investigation must localize that variability across
response phases before changing scientific implementation or policy. Do not
retry merely to select a passing sample, relax `5e-10` or the strict large-factor
gates, promote a default, or erase the earlier positive and negative receipts.
The original cold gate remains unresolved and the PR remains draft.

Ten compiled CLI tests also pass on the integrated ordinary endpoint binary.
Commit `6eb4ecd15f81e85be874cb8307afbdad000d7ce5` preserves both merge parents;
all 8,262 build-manifest entries match its source except appended historical
evidence in this note. No compiled input changes after qualification. The
merge conflict is resolved and the checked CI snapshot has no failing or
pending checks; this does not supersede the real-device numerical rejection.

### Response-boundary fingerprint instrumentation

The next diagnostic records 35 ordered, length-prefixed FP64 bit-pattern
identities at existing host boundaries: the seven retained triples cotangents,
full-Fock response, corrected Lambda, composed parameter/factor sources,
coefficient source and DF nuclear gradient, orbital solution/weights/gradient,
stationarity and final forces. Element counts distinguish actual empty
payloads. The native comparison retains only fixed scalar metadata, not copies
of these intermediates, and performs no new GPU transfer or synchronization.
Ordinary cold calls leave the private fingerprint pointer null and do not scan
numeric payloads. The existing aggregate nine-input primal census is unchanged.

Fingerprinting wall time and logical host value reads are explicit benchmark
fields; its cost remains included in diagnostic response/total timings. Hash
differences locate the first observed payload divergence, not its numerical
magnitude or cause. Equal hashes are a diagnostic census, not a cryptographic
proof or a scientific acceptance gate. The original tolerance gates remain
unchanged; fingerprints are deliberately not required to match bitwise across
schedules. This also avoids imposing a stronger numerical contract than the
qualified FP64 tolerance. The old native probe ABI is preserved beside a new
counted fingerprint-export seam.

The augmented physical water fixture independently recomputes every final-force
fingerprint from the published force bits and verifies consistent payload
lengths. All new fingerprint outputs retain their sentinels on budget-zero/one
refusal. Job 2487 passes 15 force-owner tests, one real public H2 force/energy
finite-difference test and three diagnostic memcheck cases with zero errors.
Job 2488 separately passes 45 native response, 48 independent complete-force
and 82 memcheck cases with zero errors. Local validation is 73 passes and
112 expected skips; ten integrated executable CLI tests, compiler/SCF/vendor/
default/boundary, 330-file CUDA ownership and configured formatting checks pass.

The instrumented source tree is `5dd52c7022d5b393dae456b8f66be673c99108df`.
All 8,262 manifest entries match both before and after the isolated rebuild;
this evidence append is the only later source-manifest difference. The build
records 461 ccache hits and six misses with verified explicit CXX/CUDA
launchers, checkout-root normalization and unchanged compiler/CUDA toolchains.
Library SHA256:
`c820a288fd68053a9d0025fc6f0bfb24639c585eef79466cbf31faf8d756d107`.
Comparison executable SHA256:
`deb2d44928e254bf89dd278409275fc4d6c3660052e883e32ac684a613fde9f5`.
Force-probe SHA256:
`12448964938511fde47200eb1d2091efe312c25c7d5c509168c17a2c98db4b6c`.
Receipts are `build-fingerprints/`, `fingerprints-unit-2487/` and
`fingerprints-qualification-2488/`. Job 2489 is the corresponding large
localization run and remains live at this checkpoint. Earlier completed
rejections are retained and are not superseded by this instrumentation.

### Completed response-boundary localization

Job 2489 completed normally with exit zero. Its 230-AO / 488-auxiliary
comparison uses one 24-iteration native RHF primal, identity
`466485849682549592`, and denominator identity `13715128149290587143`.
Common primal time is 306.843238 seconds and complete diagnostic time is
2213.638747 seconds. Receipts are retained in `fingerprints-large-2489/`.

| Composition | Response-only seconds | Maximum force difference against both serial references |
|---|---:|---:|
| Serial/all | 509.579518 | 2.199136429e-10 |
| Parallel/all | 443.526300 | 3.131614967e-10 |
| Omitted | 442.983783 | 2.335962535e-10 |
| Serial/all repeat | 509.621444 | 2.199136429e-10 |

All four compositions pass the unchanged `5e-10` gate in this sample.
Fingerprinting reads 98,947,403 existing host values per composition and costs
approximately 0.1083 seconds, included in the response times. Payloads 0–25
have equal identities and element counts across all four compositions:
the seven triples cotangents, full-Fock sources, corrected Lambda, all composed
parameter/factor sources, `bar_f`, and `bar_c`. The first observed differing
payload is `df_gradient` (26); every returned orbital payload (27–34) also
differs. This redirects localization toward the physical response branches,
not a presumption that gap scheduling changed their inputs.

The DF nuclear gradient is not an input to the orbital solve: that branch
independently consumes the physical reference, `bar_f`, and `bar_c`. Thus the
late orbital differences cannot simply be attributed to propagation of the
observed DF gradient difference. The returned `orbital_rhs` census is after
the second weight-map call, not necessarily the initial pre-GMRES RHS. Metric
cotangents and streamed three-center weights remain device-only and are not
fingerprinted. Equal factor and coefficient-source hashes do not establish
equality of those weights. Atomic/reduction variability and polarization
cancellation are hypotheses to investigate, not established causes.

This independently obtained cold primal and observer-instrumented sample
does not fix or supersede job 2486's same-primal omission rejection, nor the
original `3.822e-9` / `4.468e-9` cold-force failures. The gate report retains
`cold_gate_superseded: false`; no default or tolerance changes are authorized
by this passing localization sample. The next diagnostic should replay only
the two physical branches on identical captured seeds with explicit admission,
avoiding repeated Lambda solves without adding a second scientific map.

### Bounded fixed-seed physical replay

The diagnostic-only `--physical-replay` mode invokes the same complete owner
once for native RHF/DF-CCSD, triples pullback and its full-Fock companion,
and corrected Lambda. It repeats the existing DF physical source/nuclear API
four times on the unchanged factor seed, then the existing physical orbital
API four times on one final `bar_f/bar_c` seed and the original reference.
DF gradients do not feed the orbital branch. Input bit-pattern guards check
factor, reference and orbital seed lifetime; no second equations or CPU
production oracle are added. All ordinary and schedule-comparison calls keep
the private replay pointer null. Recycling and DF preconditioning are refused
for this diagnostic rather than silently changing accelerator state between
replays. Previous orbital matrices are retired before each new owner.

Small branch-gradient outputs are reserved before the cold solve. Each DF
replay explicitly admits one reusable row/metric host buffer, copies the
streamed three-center and metric weights, hashes them and discards each host
tile. This is not the earlier transfer-free host-boundary instrumentation:
new transfer bytes, buffer capacity, logical values and synchronized census
time are reported. The census includes waits for preceding producer/derivative
work, so it is not isolated transfer overhead and must not be added to its
enclosing source time. Observer timing can alter launch ordering; passing
instrumented replay is not a replacement for uninstrumented force acceptance.

Frozen source tree: `af662f96c00038c500fb3839efd345854b5a0759`.
All 8,263 manifest entries match before qualification and after the completed
large run. This historical evidence append is the only subsequent source
change. The isolated rebuild records 461 ccache hits/six misses; after updating
only the CPU forwarding test, the final qualification build records five hits
and no misses, with explicit CXX/CUDA launchers and checkout-root normalization.
Library SHA256:
`690fe1f02d749e3deb3475f0b7f9da2ce431967802c0e5a2e92aadbfaa8048c9`.
Comparison executable SHA256:
`3bcfbbf77a4a1ca61f7fed05cf24577ce4d0dae79462fb79a90b39d1579e36c0`.
Force-probe SHA256:
`38eb99c1f0bd0bd862fd52b161bfdc54f1291ea027614ac97a134f62486d16a9`.

Job 2490 passes 19 force tests (including the new independent replay force FD
and transactional budget-zero/one/577 refusal cases), the public H2 force FD,
and seven memchecks with zero errors. Job 2491 passes 45 native response,
48 independent complete-force tests across serial/parallel/omitted/default
controls, and 82 memchecks with zero errors. Local validation is 73 passes and
116 expected skips; ten compiled CLI tests, compiler/SCF/vendor/default/
electronic-boundary checks, 330-file CUDA ownership, Ruff and configured C++
formatting pass. The prior CPU forwarding mock is extended to check diagnostic
output reservation as well as unchanged reference/batch/denominator arguments.
Receipts are `build-physical/`, `build-physical-qualified/`,
`physical-unit-2490/` and `physical-qualification-2491/`.

### Completed physical replay: hash order is not error magnitude

Job 2492 completes normally, exit zero, on n2's Slurm PRO6000 allocation.
The 230-AO / 488-auxiliary diagnostic uses a 19-iteration native RHF reference,
reference identity `17213576282514911214`, factor-seed identity
`11207348646429377358`, and orbital-seed identity `1921003724709391898`.
Complete diagnostic time is 1160.512006 seconds; the single common reference,
DF source and CCSD phases are 125.452292, 3.609932 and 146.500149 seconds;
triples and Lambda take 110.657632 and 274.150179 seconds. These are different
semantic work from four complete schedule compositions and are not a cold
endpoint speedup.

| Replay | DF source/nuclear seconds | Orbital/nuclear seconds |
|---|---:|---:|
| 0 | 3.591661 | 121.035962 |
| 1 | 3.594327 | 121.378218 |
| 2 | 3.591456 | 121.454349 |
| 3 | 3.589275 | 121.510418 |

Every replay observes matching three-center and metric weight identities,
respectively `9162779198716904414` and `9505834296765261631`, as well as
matching coefficient-source identities. Each traverses 25,815,200
three-center and 238,144 metric values, transfers 208,426,752 bytes, and uses
a 1,905,152-byte census buffer. Host response-boundary fingerprinting reads
268,526 values per replay. Planned complete numeric capacity is 7,107,922,489
bytes within the unchanged 64-GiB budget; the separately sampled whole-job
device peak is 29,889 MiB over 5,799 samples, not a derivative workspace peak.

Across all six pairs, maximum DF nuclear-gradient difference is only
`4.787403112826993e-15`. Maximum orbital/nuclear-gradient difference is
`3.1174973713632426e-10`, and maximum combined branch-gradient difference is
`3.117545245394371e-10`. Returned RHS, Z solution, hcore/overlap/Fock weights,
stationarity and orbital gradient hashes differ in every pair. All orbital
calls take 12 iterations/13 operator actions; maximum independently audited
Z residual is `1.3688750763316764e-13` and maximum stationarity is
`8.255108402410727e-12`. The observed combined variability passes `5e-10`
against both endpoint replay references, but these are independently replayed
branches with one final coefficient seed, not four cold force endpoints.
Raw JSON, all-pairs report, trace, memory samples, input/native hashes and
source/reproduction receipts are retained in `physical-large-2492/`.

This is the important correction to the earlier localization: the first
differing hash is not the dominant numerical error. In this observed run,
the streamed DF weights agree and DF nuclear contraction variability is
approximately five orders of magnitude below orbital/nuclear variability.
Changing DF atomic accumulation solely because its hash differed first would
therefore target the wrong branch. The next localization must distinguish
initial orbital RHS/Z variability from the one-/two-electron nuclear pieces
and their polarization contraction; returned final RHS hashes alone cannot
make that distinction. Atomic J/K/reduction ordering and polarization
cancellation remain hypotheses, not proved causes.

This passing, different cold reference and weight-observed replay does not
supersede job 2486's omission rejection or the original cold-force failures.
No tolerance, screening, residual, precision, or default policy changes are
made. PR #1999 remains draft and the complete-consumer promotion remains
blocked by the original scientific gates, not completed by diagnostics.

### Remaining gates

The same-primal diagnostic adds a private native-state replay input to the existing
complete force owner, not a second composition. The only caller is a new
same-primal comparison entry that obtains its own native CUDA cold state,
admits pinned/working host copies before allocation, and shares the exact
original source/frame. Four schedules include a final serial repeat so response
variability can be distinguished from demand/reduction changes. Output buffers
are fixed and charged before the common solve; original host copies remain
reserved through all response phases. Shared DF ownership is added back to the
late orbital phase after the working CC state is consumed. A still-live native
exact-reference source is also explicitly charged beside that phase's new
orbital provider. The ordinary cold owner retains its scientific equations and
defaults, with this live-source allowance corrected rather than ignored.
Comparison timings include common-owner retirement and must not be called cold
endpoint speedups. Small diagnostic gates and the earlier large comparison
pass, but the integrated large omission comparison above fails. Neither that
rejection nor the original cold failures is superseded.

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
