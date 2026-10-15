# Investigation: r2SCAN branch reuse, contraction and complete-grid totals

Status: rejected; global SCF CSE and serial compensated-XC prototype not promoted
Date: 2026-10-15

## Problem and retained boundary

The historical #2464 r2SCAN-3c direct/96 moved endpoint changed from
296.814757 to 403.782686 seconds with 14 to 23 Fock builds. The stationary
force itself improved. At candidate iteration 14 the residual was already
1.313e-13, but its compensated electronic energy change was 5.061e-12,
above the unchanged 1e-12 gate. That observation is real, but one pair does
not establish a repeatable CSE-specific SCF penalty. #2507 owns the follow-up.

Current experimental control is master
`5f80ba629b5f20412215cdc690f8fa458ac3fdf8`. The initial candidate changes
only FP64 semilocal equal-spin lazy-branch sharing. Grid, basis, method,
precision, screening and stopping gates are unchanged. The original working
directory's unrelated changes are not part of this experiment.

The global semilocal SCF CSE switch is withdrawn. Its two original experimental
sources/libraries remain immutable. A separate serial compensated-XC trial is
also rejected: it does not eliminate energy-gate tails and adds complete-endpoint
cost at equal work. Neither experimental production-source change remains in
this worktree. Consumer-scoped stationary-force CSE is owned and qualified
independently in #2520. No production SCF scheduling default, stopping threshold,
FMA policy, or MD admission cap is changed by this investigation.

## Independent molecular trajectory observations

The separate #2507 driver uses one native original-geometry density snapshot,
restored and re-exported bitwise in every fresh moved owner. Three paired
direct/96 reconvergences pass all six independent 1e-8 Eh / 1e-7 Eh/Bohr gates:

| Pair | Control Fock builds | CSE Fock builds | Control E+F (s) | CSE E+F (s) |
| --- | ---: | ---: | ---: | ---: |
| 0 | 16 | 14 | 313.174577 | 278.028051 |
| 1 | 14 | 17 | 290.512189 | 324.843002 |
| 2 | 15 | 15 | 298.313844 | 289.828171 |

Pair 1 retains a real +11.8% complete-endpoint regression with three extra
Fock builds. Neither that observation nor the historical +9-build penalty is
discarded or normalized by iteration count. The mixed signs do not establish
a consistent CSE-specific SCF penalty.

Slurm 8535 then holds the exact source, library, GPU (node1, visible 0), and
native seed fixed. Two fresh control-only reconvergences take 13 and 14 Fock
builds (271.433950 and 285.151632 seconds); both pass independent E/F gates.
At the latter's iteration 13 the residual is 1.561e-13 but its energy change
is 1.176e-12, so the unchanged energy gate requires another physical build.
The first observed component difference is at evaluation 2: one-electron
-3.638e-12 Eh, Hartree +5.457e-12 Eh, with XC equal at that step. This proves
real same-GPU incumbent trajectory variability, **not** a particular Fock
kernel's responsibility or timing noise, and not a unique XC root cause.

The seed snapshot SHA256 is
`84cb007bfbb43233565a782d6533ef87209a0fc04b2f408482d2f3d43944b7d1`;
its density SHA256 is
`8ea694fd03c3716b6aa1d2dbd6c388d787e1a9c1d2e4744fa208521e97289a8a`.
The original-geometry native seed takes 27 builds. An earlier independent
34-build cold solve passes science gates but its subsequent public checkpoint
export fails because r2SCAN-3c has no resolved accuracy model for that API.
Both attempts are retained; different cold seeds are not same-seed moved
replicates. The benchmark-only bounded native warm-state ABI avoids changing
production checkpoint compatibility to accommodate the experiment.

## Scalar rounding probe and the production-policy distinction

The earlier 256-point bitwise comparison is not a universal CUDA contract.
Separate processes emit each current source's production r2SCAN equal-spin
body. A deterministic physical input census contains 262,228 points across
density/gradient/kinetic scales and work-boundary neighbors, seed 2507.

On n1/sm_120, CUDA 12.9, Slurm 8508, ordinary `-O3` produces different bits
at 138,199 points; energy differs at 51,212 points. All 2,097,824 output
values per arm are finite. These are A/B scalar observations, not an
independent oracle or molecular force-error measurement.

With the same inputs and sources, compiling **both** probes with
`--fmad=false` makes all eight output channels bitwise identical at every
point (Slurm 8509). This isolates contraction-dependent rounding in that
standalone scalar schedule, **not the cause of the production regression**.

A subsequent audit of both actual CMake compile-command receipts found that
`generated_grid_policy.cu` already uses `--fmad=false`. The policy protects
r2SCAN minority-spin boundaries and is also present in historical control
`0e6dfb453` and candidate `786c31a48`. The compiler-side CUDA grid adapter
retains the same policy. The ordinary-FMA probe therefore exercises a different
arithmetic contract; its differences must not be presented as production SCF
differences. The no-FMA census is positive parity evidence for the actual point
contract, but not a universal bitwise or full-molecule guarantee. No production
FMA flag or stopping tolerance is changed by this investigation.

Raw inputs, outputs, source, libraries, compiler/cache receipts and comparisons
are retained under `/data/jzzeng/qc-r2scan-cse-20261015/point-probes` and
`point-probes-no-fma` on n1. An initial missing-NVRTC environment failure is
retained separately; setting the CUDA library search path fixes the harness.
`production-point-compile-policy.json` retains both actual protected production
commands separately from those standalone probes.

The installed production libraries also retain a `cuobjdump` resource census.
Both r2SCAN equal-spin point variants use 255 registers in both arms. The
single-tile stack frame changes from 200 to 216 bytes and the batched frame
from 224 to 240 bytes. These are compiled-resource observations, not endpoint
timings or a claim that CSE necessarily reduces register pressure. The compact
comparison is `point-resource-usage.json`; full disassembly-tool receipts remain
in each arm's build evidence.

## Complete-grid energy reduction hypothesis

The 96-atom protocol has 2,359,296 points and 256-point tiles: 9,216 tile
totals enter the original ordinary FP64 serial accumulation. The tiled,
compact-batch and fallback consumers all perform ordinary point/tile sums.
The KS electronic-energy delta already retains compensated matrix traces,
but receives XC energy **after** that complete-grid rounding. Compensation
of matrix traces cannot recover low words already lost by XC integration.

This was a possible source of instability at a strict 1e-12 SCF energy gate,
not evidence of CSE-specific point differences or proof that it explains the
molecular tail. The numerical trial is independent of the scalar CSE trial.
It tested a generic ordered Neumaier reduction to FP64 XC integration, retaining
three low words across batches and folding once at publication. The point
program and potential matrix are unchanged. The owner charges exactly three
extra doubles (24 bytes) through its shared overflow-checked arena layout;
there is no full-grid retained array, hidden allocation or CPU reference path.
FP32 contraction schedules retain their existing reduction.

Four source-matched arms distinguish incumbent/CSE from compensated/
compensated+CSE. The original two sources and libraries remain immutable.
Independent `math.fsum` and exact dyadic cancellation tests cover all three
channels, tile/batch tails, the full grid census, and nonfinite rejection.
All 23 initial real-GPU reduction tests pass; this is not molecular admission.

Slurm 8515 also passes the independent Libxc r2SCAN boundary checks (25 total
tests) and a four-case low-word/nonfinite memcheck subset with zero errors.
Slurm 8519 and the final source-bound Slurm 8524 exercise actual native XC
integration, including dense,
indexed, compact/ragged, optional-provider fallbacks, retained FP32 schedules,
response, and graph replay. Exact-sized component arenas execute successfully;
one-byte-short arenas fail closed for both spin layouts of all five curated
functionals. The complete native suite and three native memcheck subsets pass
with zero memory errors. Raw compiler, executable and scheduler receipts are
retained under `native-test-evidence` on n1. Native integration success does
not replace independent complete-molecule gates.

## Why serial compensation is not the production fix

Slurm 8542 reuses the exact original native checkpoint above, with a fresh owner
for each independent moved reconvergence. Compensation alone takes 13 Fock builds
and 286.814613 seconds; compensation plus global SCF CSE takes 15 builds and
306.547579 seconds. Both pass the original independent gates, with maximum
energy/force errors 3.4561e-11 Eh and 1.2217e-10 Eh/Bohr. These are one diagnostic
pair, not three reconvergence replicates or molecular performance admission.

The compensated+CSE solve still cannot stop at iteration 13: energy change is
1.8731e-12 while residual is 1.5914e-13 and density change is 1.3850e-12. At
iteration 14 its energy change remains 1.6919e-12, above the unchanged gate;
iteration 15 finally reaches 1.9942e-14. Compensating this XC reduction does not
eliminate the observed energy-only blocking mechanism. It does not prove that
another energy component is the unique origin either.

A separate complete-endpoint campaign compares compensation alone with incumbent
SCF arithmetic. Each pair shares one Slurm GPU, with alternating arm order and
an unchanged 1800-second bound per arm. Each arm requests cold, five frozen-warm,
moved and five moved-warm E+analytic-force calls. All returned energies and force
arrays are independently rechecked against the exact retained geometry/protocol
oracles; reference timings are not new measurements. At matched work, the serial
per-channel reduction adds cost rather than admitting the rejected global CSE.
Slurm 8543-8546 return 95 of 96 requested calls. All 95 independently pass,
with maximum energy/force errors 4.0473e-11 Eh and 1.3865e-10 Eh/Bohr.
Seven populations complete all 12 calls. The incumbent direct/96 population
hits its original 1800-second limit during its fifth moved-warm call; only
four of five return, so there is no moved-warm median or complete-population
admission for that arm. No timeout is retried or replaced.

Table entries are complete cold/warm/moved/moved-warm seconds, with all five
warm samples used when complete. The last two columns give cold/moved Fock
counts; every warm call has one build except the explicitly retained case below.

| Provider | Atoms | Incumbent seconds | Compensated seconds | Incumbent Focks | Compensated Focks |
| --- | ---: | --- | --- | --- | --- |
| direct | 48 | 83.040/22.064/61.797/22.202 | 86.209/22.389/64.011/22.473 | 19/13 | 19/13 |
| DF | 48 | 50.822/6.313/36.853/6.326 | 53.759/6.456/38.763/6.452 | 19/13 | 19/13 |
| direct | 96 | 627.526/95.198/293.361/incomplete 4 of 5 | 478.685/94.783/282.425/94.858 | 35/14 | 26/13 |
| DF | 96 | 208.122/22.403/133.287/22.545 | 210.054/22.746/137.436/22.687 | 25/13 | 24/13 |

The matched-work direct/48, DF/48 and DF/96 warm medians regress by roughly
1.47%, 2.27% and 1.53%. Do not use direct/96's lower cold time as evidence
for the reduction: its SCF work changes from 35 to 26 builds. Its fourth
compensated frozen-warm call also takes two builds and 109.012978 seconds;
the complete five-sample median retains it. Serial compensation is neither
a repeatable SCF-tail cure nor an equal-work endpoint optimization.

All timing ranges, raw energy/force arrays, iteration histories, outcomes and
source/library identities remain under `endpoints/xc-totals-1` on n1 and in
the local ignored `remote-endpoints/xc-totals-1` copy. The independent audit
archive SHA256 is
`39e260c376310dd8acc2d9c82f8b0be066f41e47729d7098bd51d7d7abd6e45f`;
the separate two-record same-seed numerical audit SHA256 is
`6a445bea85180edb650350e5f995ae72e8731bc87b55bbb9426b4806c39cef4e`.

The complete eight-file prototype, including the otherwise untracked helper and
GPU tests, is preserved as `xc-totals-prototype.patch`, SHA256
`28dd428b143468aaae309cd754e9740ec033280f2002e73784cc23fbd128b3ea`.
Its reverse applicability was checked against the measured local source before
removal. Both numerical source/library arms remain immutable on n1. The local
final tree has no production compiler, runtime or test changes from this trial.

## Rejected diagnosis and future admission gates

MD primitive/pair compaction and reciprocal floating atomics are real sources
of arrival-order variation, but the full 96-atom descriptor estimate exceeds
the existing 128 MiB optional MD cap. Do not assert this optional provider is
the cause without observing actual admission/work; do not raise its cap as
an unrelated workaround. The original XC matrix/totals kernels use ordered
stream/tile traversal, not floating atomics.

A discarded standalone-J diagnostic illustrates another consumer boundary.
The public `FockPlan` preparation reports a capable MD layout at 12/48 atoms
and a generated-shell/canonical fallback layout at 96 atoms (Slurm 8541,
explicit 1 GiB component budget). Its host-array `evaluate` route, however,
calls `execute_cuda_direct_jk_range`, which directly launches the independent
AO-quartet kernel rather than the KS device enqueue/MD dispatch. A 48-atom
attempt times out at its diagnostic 540-second bound (Slurm 8532); a smaller
12-atom attempt returns two identical J matrices and is then explicitly
stopped (Slurm 8538). Neither produces MD work-counter output. These are not
MD-positive execution controls, full KS admission observations, or molecular
performance results. Preserve the setup/replay distinction; do not repeatedly
run this much slower public diagnostic as a proxy for the production KS route.
The original scripts, partial arrays and explicit timeout/operator-stop outcomes
remain under `j-diagnostics` on n1. No MD source or admission limit is changed.

Future SCF numerical/scheduling trials must reuse one native original-geometry
checkpoint, retain its density/blob hashes and every iteration, and verify the
exact reference protocol and both-geometry independent E/F gates. Frozen warm
repeats are not reconvergence replicates. Require replicated reconvergences and
complete direct/DF 48/96 cold + five warm + moved + five moved-warm endpoints
before promoting a global SCF switch. Retain every negative and incomplete
population and the 1800-second bound. A better bounded reduction may be revisited
only with independent numerical coverage and demonstrated complete-endpoint
benefit; neither mathematical accuracy alone nor changing a stopping tolerance
is performance admission.

The independently owned force-only alternative opts in only for the FP64
stationary-force point program, leaving CPU, semilocal SCF and FP32 point source
unchanged. Its original qualification covers 192 independently accepted native
calls over r2SCAN/r2SCAN-3c, direct/DF and 12/48 atoms. It retains small cold
negatives rather than extrapolating a point-kernel gain. The separate 96-atom
extension uses those same immutable core/SPD libraries, its own output/cache
directories and the same complete-endpoint gates. It is not a numerical arm of
the compensation trial, and unmatched-work endpoint differences cannot be
attributed to force CSE. #2520 owns the production patch; this investigation
does not duplicate it or declare the SCF-tail issue fixed.

## Completed 96-atom force-only extension

Slurm 8553-8556 complete all eight requested populations: 96 of 96 E+analytic-
force calls independently pass the original 1e-8 Eh / 1e-7 Eh/Bohr gates.
Maximum energy/force errors are 2.1965e-10 Eh and 1.5652e-10 Eh/Bohr. Each pair
uses one allocation/GPU, with five warm and five moved-warm calls and the
unchanged 1800-second limit per arm. The paired source and oracle protocols,
core libraries, SPD manifests/DSOs and every returned array/history are retained.
Final source-bound reduction runs only after all GPU populations finish.

These are the separately owned immutable sources/libraries under
`/data/jzzeng/qc-r2scan-force-branches-20261015/{control,candidate}`. No sources,
libraries or existing qualification outputs in that campaign are modified.
Own outputs and optional runtime caches live under this investigation's n1 root.
Core library SHA256 values are:

```text
control   17e24ba1f7f015cc667c67748643aaa4e48979159b7576ca8393a67b8b9e8b41
candidate 5b70a2397833f3f6c74e610c62d167d54161414f9d557b5a3ae9de508badf09c
```

The candidate is the independently retained scientific patch SHA256
`d8e40a5c38bf978f9e9814b8bbcf8dd5ed5f6ebe2e3c0c942a374ace814f1d98`
on the same pinned master. Table entries again give complete seconds in
cold/warm/moved/moved-warm order, not kernel-only time or work-normalized time.

| Method | Provider | Control seconds | Force-CSE seconds | Control cold/moved Focks | Candidate cold/moved Focks |
| --- | --- | --- | --- | --- | --- |
| r2SCAN | direct | 115.484/17.031/109.905/17.095 | 114.984/16.584/82.556/16.411 | 19/17 | 19/12 |
| r2SCAN | DF | 107.700/14.352/79.981/14.317 | 106.796/13.361/79.055/13.351 | 18/12 | 18/12 |
| r2SCAN-3c | direct | 459.939/91.234/354.766/91.225 | 430.366/90.519/423.968/90.950 | 26/19 | 24/24 |
| r2SCAN-3c | DF | 208.266/21.979/131.868/21.622 | 205.188/20.480/273.798/19.861 | 25/13 | 25/13 |

Every warm/moved-warm call takes one Fock build. Matched-work warm medians
improve by 2.63%/6.91% for direct/DF r2SCAN and 0.78%/6.82% for direct/DF
r2SCAN-3c. Moved-warm medians improve by 4.00%/6.74% and 0.30%/8.14%, respectively.
The small direct-3c differences are not a claim of a large integral-path win.

Two moved negatives prevent an unconditional all-phase 96-atom performance
admission. Direct r2SCAN-3c regresses by 19.51%, with 19 to 24 Fock builds.
Conversely direct r2SCAN's moved improvement includes 17 to 12 builds. Neither
change can be attributed to a faster/slower force point program: the SCF source
is preserved and the incumbent's independent same-seed variation is already
documented. Report these as unmatched-work endpoint observations, not force-CSE
speedups or a proven SCF scheduling effect.

DF r2SCAN-3c regresses by 107.63% at matched 13-build work. Its stationary-force
telemetry locates the excess wall time in `prepared_owner_lookup_or_construction`:
0.711902 to 144.350203 seconds. XC geometry time improves from 7.549308 to
6.556275 seconds, and stationary integral derivatives remain 2.613762 versus
2.608072 seconds. The aggregate `compile_aot_cache_setup` category is not proof
of a compilation: both own runtime-cache directories remain empty on this
AOT/native route. The prepared owner also verifies full artifact hashes before
geometry rebind; core images are approximately 0.9 GiB. Host-only partial
integrity audits overlap this campaign and themselves experience multi-minute
I/O waits. No subphase trace distinguishes hash I/O, owner rebind or other work,
so do not label this uniquely as a JIT, hashing, infrastructure or CSE regression.
Keep the complete negative; do not subtract the stall or weaken real artifact
integrity checks to manufacture an admitted endpoint.

This extends scientific coverage to 96 atoms and supports the bounded force-only
direction. It does not close #2507's complete all-phase promotion requirement or
establish a root fix for SCF tails. #2520 retains ownership of the production
patch and its original 12/48 performance claims; all complete 96-atom results,
including both negatives, are handed back to that task without a duplicate PR.

The source-bound 96-atom audit archive is
`endpoints/force-only-96/audited-endpoints.json.xz`, SHA256
`ed4d6a1b18a9de19c8b26ab2c812d1585499c7dc2c1e5877f18d8597be6f1509`.
Its local ignored copy is under `remote-endpoints/force-only-96`; `timings.md`
retains all sample ranges and work pairs, and `force96-provenance` binds the
core/SPD artifacts. The original runners are retained as `run-force96-*.sh`.
Reproduction uses finite `srun` through `run-force96-matrix.sh`, preserves Slurm
device visibility, verifies the compiler-cache executable, and requires a fresh
output campaign rather than overwriting any of these measured populations.

After withdrawing both numerical production trials, the final local compiler
inventory has 540 modules and zero dependency errors. Focused semilocal native
and FP32 generation checks pass 35 tests with 65 explicitly optional cases
skipped. The archived eight-file prototype applies cleanly to the restored base;
the final compiler/runtime/test tree has no changes from that base.

## References

The subsequent actual-KS fixed-density replay and first-iteration component
snapshots locate the first divergence in the generic Direct-J floating scatter,
not the XC point program. Its targeted numerical control and qualification
limits are recorded in
`../proposed/2026-10-15-r2scan-direct-j-roundoff-root-cause.md`. This extends the
diagnosis without promoting either rejected numerical trial in this note.

- #2464, #2507, #2520, and #2511's consumer-scoped force arithmetic precedent.
- `.agents/notes/implemented/performance/2026-10-14-xc-shared-branch-regions.md`.
- `.artifacts/r2scan-semilo-cse-20261015` in the reused PR worktree.
- Immutable experiment trees, build receipts and raw outputs on n1:
  `/data/jzzeng/qc-r2scan-cse-20261015`.
- Final independent #2507 evidence SHA256:
  `7457b3372d4c8d56023679ef8a2a2414ede360f36dc2079c272303a77fdbbf1f`.
- Independent reconvergence driver: `benchmarks/experiments/issue2507_semilocal_cse`
  in the separate #2507 task; evidence on n1 under
  `/data/jzzeng/qc-2507-reconvergence-20261015`.

## Conditional follow-up (2026-10-15)

The [coupled J/XC decision](../implemented/numerics/2026-10-15-coupled-j-xc-compensation.md)
holds the independently diagnosed J correction fixed before testing XC scalar
compensation. That is a different control from the XC-only trial rejected here:
its serial version removes energy-only cold tails, and its bounded warp reducer
addresses the serial version's fixed-work cost. Preserve this note's historical
negative; neither new control enables global SCF CSE or attributes the original
first divergence to XC.
