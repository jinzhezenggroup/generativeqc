# Decision: retain three-index unrestricted DF factors and reverse them directly

Status: implemented
Date: 2026-11-04

## Problem

The initial bounded batching change for #2978–#2980 is preserved in
[the earlier note](2026-11-04-unrestricted-admission-and-force-batching.md).
It fixes singleton UMP2 force requests, but reconstructing the fitted interaction
as an AO fourth-order tensor still defeats DF storage scaling. Repeating a
whole-AO transform for every orbital-gradient slice is unnecessary even when
each individual request fits the memory budget.

## Decision

The immutable fitted source owns raw three-center integrals, the native FP64
metric inverse square root, AO three-index factors, and alpha/beta MO
three-index factors. The compiler owns the staged two-orbital transformation,
rectangular pair-panel Gram contraction and complete reverse program. The
reverse includes both coefficient legs, raw three-center and metric-root
cotangents. The existing fixed-rank spectral derivative and nuclear integral
owners complete the physical response; no independent-reference oracle runs
inside a production endpoint.

The unused dense-AO native evaluation/pullback interface and its generated
buffers are retired rather than kept as a compatibility shim. The small
interpreted AO Gram algebra remains an independent compiler test alongside the
new native representation; production generation emits only three-index maps
and rectangular blocks.

The native provider verifies the accepted physical orbital frame before
replaying factors. Ordered same-spin pair requests reconstruct only their
rectangular MO block, not an AO interaction tensor. UMP2 constructs initial
weights by occupied pairs and virtual-pair blocks, including exchange by
permutation. Structural force weights accumulate both Gram-leg cotangents
directly into spin factors. Their complete generated VJP supplies the orbital
gradient and the final raw/root nuclear cotangents. Full DF UMP2 therefore has
neither an AO fourth-order value nor an AO fourth-order force cotangent.

The CUDA owner borrows immutable host factors and optionally owns device factor
copies. A refused optional allocation falls back to admitted selected-panel
uploads and the same generated CUDA Gram contraction, never host contraction.
Every published result follows stream completion; shared execution is serialized
and device allocations carry stream-aware cleanup. Capacity admission includes
retained factors, simultaneous panels, output copies, selection indices,
generated forward/reverse arenas and backend nuclear-response staging.

UCC fitted nuclear response also reverses factor cotangents without an AO
fourth-order transform. UCC's generated MO tensors, Lambda equations and ordered
triples evaluation are distinct costs, not removed by this source change. The
finite shared work policy remains a necessary resource boundary. Progress
scopes separate graph programs, Lambda, triples, orbital and nuclear response
so complete-endpoint costs can be measured rather than inferred from RCC or
integral microbenchmarks.

## Rejected alternatives

- Raising a scalar-work limit cannot repair repeated transforms or storage.
- A full spin-channel MO value is a useful bounded conventional fallback, not
  the required resident representation of a fitted interaction.
- A permanently resident device cache would turn an optional performance
  optimization into a correctness/resource precondition.
- Ignoring either coefficient leg or the metric/nuclear derivative produces
  an incomplete force even if unrelaxed energy integrals agree.

## Invariants and limits

Retain exact frame/backend ownership, checked arithmetic, finite outputs,
complete output publication, independent scientific gates and the conventional
bounded slice fallback. Correlation-only DF still owns a conventional reference
response explicitly. Native host FP64 metric conditioning and detached
factor/cotangent transport are intentional hybrid work, not a silent backend
substitution. No broad production-size scalability or RCC/UCC speed ratio is
claimed. The qualified finite physical ladder must not be extrapolated to the
issue's hundreds-of-AO workload.

## Evidence

The compiler test compares forward factors and rectangular Gram blocks against
independent NumPy algebra, and checks directional VJPs separately for raw
integrals, the root and both physical spin frames. Native contracts compare
requested fitted blocks to an independent dense contraction, reject stale
frames, and exercise resource admission. The opt-in CUDA contract compares
resident and zero-residency panel execution across spin channels, padded slots
and zero-beta factors. Physical endpoint qualification uses independent PySCF
energies and two-step reconverged force differences, never native finite
differences as their own oracle.

### Implementation-base qualification (historical)

The implementation-base compact record at commit
`365915f59ea7e53f5da39af9784626707125addf` retains numerical gates, all endpoint
timing populations, capacity, source-reuse,
transfer and separated response-phase results. Its column-labelled arrays use
H2/H3/H4 row order, seconds rounded to six decimals, and per-endpoint phase/work
counts; force errors correspond to steps 1e-3 and 3e-4 Bohr. Phase times are
instrumented except the first endpoint's primal CC/(T) metadata. It also binds
the full summary and provenance manifest by SHA-256. The full summary includes
native forces, independent reconverged directional derivatives and individual
generated graph action counts. The manifest binds input snapshots, source
inventory, executed library, build/cache logs and compressed raw traces. Full
artifacts remain in the ignored local `build/issues-2978-2980-factor-evidence/`
directory and source-matched n1 checkout. This retains a compact accepted record
within the repository's evidence budget, not raw traces in Git or a new external
publication.

All 2076 scientific inputs match implementation commit
`8316b370cc092d3a9675d939deab5ec289966e2e` on base
`c776b6cc12855b68195e4be27d0604b1b524690e`, with embedded input identity
`c416b12e1a28d1fd10392670b692c78fdada284d063c6b54a3f340cf3a3913df`.
Executed CUDA library SHA-256:
`81efdc6026c6e8b3c25048be0e3ec551b9c3affc02babc6ba0c026cc81d4fd81`.
The qualification checkout is the base plus byte-matched implementation inputs,
not a claim that its Git HEAD is the implementation commit. Subsequent catalog,
test-harness and evidence changes do not change the scientific input identity.

The complete-library Slurm run on RTX 5090 passed resident and bounded-panel
execution, eight compact/physical CUDA force cases, four original Br RHF/MP2
energy/force cases and the full physical H2/H3/H4 DZ/TZ CBS ladder. Separate
PySCF 2.14.0 processes passed the complete CBS energy gate of 3e-8 Eh and both
independent directional-force gates of 5e-7 Eh/Bohr. Trace modes and repeated
endpoints agree. These are complete new-library observations, not the historical
intermediate patch's timings or a mixed-object diagnostic executable.

The measured native H4 endpoint takes about 350 s untraced; its single traced
endpoint takes about 451 s, of which the triples-response parent scope accounts
for 430 s. The virtual-axis `moment_aaa` programs each execute over a million
observed ordered graph actions. This is a demonstrated residual UCC response
cost, not the removed fitted UMP2 singleton-transform problem. Preserve the
finite work policy and do not advertise large-system scalability or an L20/RCC
speed ratio from the fitted-source improvement. Detailed tracing materially
perturbs this execution schedule; phase times are nested and are not additive
with complete endpoint timings.

The implementation-base CI exercise also exposed extracted host probes missing
newly required SCF/exception dependencies and the optional fitted-planner boundary. Updating
those numerical test doubles preserves their status/lifetime controls without
weakening production admission or changing the scientific source. Full GCC and
Clang native CI passed on the same scientific inputs; the targeted updated
controls pass locally. The Libxc snapshot refresh updates checked input
identities only: its stale historical measurements and all capability stages
remain unchanged.

### Implementation-base measurement detail and interpretation

Qualification used RTX 5090 (32607 MiB), driver 580.95.05, Release, GCC 13.3.0,
nvcc 12.9.86 from the CUDA 12.9.1 installation, `sm_120`, verified sccache 0.16.0,
scalar host linear algebra and one native host thread. AOT shell/stationary
artifacts were disabled. Shared before/after cache statistics cannot establish
exclusive cache-hit deltas for this build. Slurm job 15846 used `main`, node1,
`--gres=gpu:5090:1`, a finite one-hour allocation and unchanged scheduler device
visibility. The harness set no scalar-work override.

Full CBS requested energy and analytic forces with spherical cc-pV(D,T)Z/JKFIT,
full DF-HF, `standard`, `preconvergence="auto"`, 32 GiB numeric budget and SCF
ceiling 200. H2/H3 use the original #2980 input and basis snapshots; H4 uses the
physical triplet ladder geometry and exact PySCF 2.14.0 basis snapshots. No
reference oracle runs inside a production endpoint. Executed CUDA backend does
not imply that host orchestration, metric conditioning, factor transport or
every UCC provider contraction is device resident.

Setup includes local input/basis reads and calculator construction, not imports
or compilation. Warm calls compute the requested complete endpoint again. H2/H3
have three endpoints per trace mode; H4 has two trace-off and one trace-on call,
so H4 has no trace-on warm median.

| Case | Setup + first, trace off (s) | Warm median, trace off (s) | Warm median, trace on (s) | Sampled device peak (MiB) | Maximum planned numeric bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| H2 triplet | 1.45928 | 0.18745 | 0.28195 | 6175 | 11,594,937,039 |
| H3 doublet | 20.29199 | 18.94008 | 25.08417 | 6207 | 11,864,888,440 |
| H4 triplet | 350.79100 | 349.97163 | not measured | 6263 | 12,486,832,661 |

Device peaks are 200 ms whole-device `nvidia-smi memory.used` samples, not
allocator-exact or process-RSS peaks. Planned numeric capacity is a separate
conservative admission quantity. All repeats have identical reported energies;
their largest force-component difference is below 9e-16 Eh/Bohr.

| Case | Maximum independent energy error (Eh) | Maximum checked directional-force error (Eh/Bohr) |
| --- | ---: | ---: |
| H2 | 5.01e-12 | 7.67e-8 |
| H3 | 2.99e-11 | 2.37e-8 |
| H4 | 2.67e-11 | 2.80e-7 |

These UMP2 initial-weight counters cover both cardinal components of one CBS
endpoint, excluding factor preparation, pullbacks and other endpoint transfers.
They are semantic work, not measured FLOPs or kernel counts. Zero provider AO
scans means reuse after factor preparation, not zero initial integral work.

| Case | Factor-block reuses | Provider AO scans/values | `transform_fmas` | H2D bytes | D2H bytes |
| --- | ---: | --- | ---: | ---: | ---: |
| H2 | 8 | 0 / 0 | 174,016 | 2,304 | 23,712 |
| H3 | 14 | 0 / 0 | 1,107,774 | 6,224 | 100,832 |
| H4 | 26 | 0 / 0 | 4,807,772 | 15,136 | 327,960 |

The following seconds are per traced CBS endpoint, averaged over its recorded
population, except primal solver/(T) metadata from the first untraced endpoint.
Nested graph durations must not be added to their parent response phase.

| Case | Primal CC solver | Primal (T) | Lambda | Triples response | Orbital response | Nuclear response |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H2 | 0.01294 | 0.000004 | 0.03333 | 0.000011 | 0.00347 | 0.00556 |
| H3 | 0.07225 | 1.03521 | 0.78909 | 22.54484 | 0.02564 | 0.01673 |
| H4 | 0.11039 | 15.72394 | 1.78043 | 430.33813 | 0.02188 | 0.04016 |

H4's three `moment_aaa` virtual-axis programs each execute 1,193,859 observed
graph actions. UMP2 initial weights across both bases take
0.00116/0.00266/0.00624 s for H2/H3/H4 in these traces; orbital-gradient scopes
take 0.00867/0.01356/0.04147 s. This distinguishes the remaining UCC response
schedule from the removed UMP2 singleton/full-axis source-transform cost.

CPU scientific selection passed 74 tests with eight device cases skipped;
catalog tests passed 63; additional SCF/status/retirement controls passed 74.
The four original Br cases passed complete native CUDA RHF/MP2 energies and
forces with both independent displacement steps in 133.65 s. The three-case
physical ladder suite took 1007.24 s including independent oracle work; that
suite duration is not a native endpoint latency. Permanent real-device gates
can be repeated after configuring the documented cached CUDA build:

```bash
srun --partition=main --nodelist=node1 --gres=gpu:5090:1 \
  --nodes=1 --ntasks=1 --time=00:45:00 bash -lc '
    export GENERATIVEQC_LIBRARY="$PWD/build/cbs-open-shell-cuda/libgenerativeqc.so"
    export PYTHONPATH="$PWD/python:$PWD:$PWD/tests/python"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    export GENERATIVEQC_OPEN_SHELL_CUDA_TEST=1
    export GENERATIVEQC_OPEN_SHELL_CUDA_LADDER_TEST=1
    export GENERATIVEQC_FULL_DF_CUDA_TEST=1
    build/cbs-open-shell-cuda/generativeqc_df_factor_cuda_tests &&
    python -m pytest -q tests/python/test_cbs_open_shell_native.py \
      -k "(df_analytic_components and cuda) or physical_hydrogen_df_ump2_force or physical_df_cbs_size_ladder" &&
    python -m pytest -q tests/python/test_df_bromine_reference.py
  '
```

### Final post-native-policy integration qualification

The accepted record now refers to the combined upstream native-policy/CBS
implementation at base `ffa6936968d5c5a33194bd9bc09435a9f2ee0d05` and the
review follow-up `174ebaa936321ff4dedc2804c2981c0a05e5b9cb`, not the historical
2076-input population above. All **2085 scientific inputs** match the final
qualification checkout. Both the CPU library and the CUDA library loaded
inside Slurm report source identity
`e73041c5d9cfaf573e4dbbfe17e7fdf317457ab15b54b724834876d844791554`.
Executed CUDA library SHA-256:
`7bd78bb3e65eb2d88ed900f4548358d71d72271be2be017d575e0169b42f1fd0`.
The remote checkout's Git HEAD is not asserted to be the PR commit; the complete
input inventory and embedded library queries establish the source match.

The follow-up admits both fitted Gram cotangent legs cumulatively before the
UCC nuclear-response seed traversal, reports the actual 300-iteration Lambda
GMRES ceiling, and preserves source-indexed UHF device JK/allocation failure
details. The executable response control narrows a finite policy after
preflight, exercises Lambda and final fitted-seed refusals, and verifies exact
force recovery after refusal. Four compiled UHF failure-injection controls pass;
all four correctly fail against the pre-fix parent implementation.

Final-source local qualification passes **121/121 configured native CPU tests**
and **363 Python tests**, with eight device cases skipped and 70 cases
deselected in that CPU selection. All applicable pre-commit checks pass. Slurm
job **15853**, `main`, node1, `gpu:5090:1`, uses the same verified Release/GCC
13.3.0/nvcc 12.9.86/sccache 0.16.0 toolchain, an explicit one-hour limit and
unchanged scheduler visibility. Its captured runtime environment confirms one
host thread and no scalar-work override. It passes the native resident and
zero-residency panel contract, eight compact/physical CUDA force tests (74.25 s),
and all four original Br RHF/MP2 energy/force cases (132.74 s).

All **15 complete CBS energy/analytic-force endpoints** finish on this final
library: three trace-off and three trace-on calls each for original H2/H3, and
two trace-off plus one trace-on call for physical H4. Fresh separate PySCF
2.14.0 calculations pass the matched full-CBS energy and both reconverged
directional-force gates. The duplicate three-case `physical_df_cbs_size_ladder`
pytest block was not rerun on this final source; the complete receipt population
and fresh independent oracle qualify that ladder instead. Its earlier pytest
pass belongs only to the implementation-base population documented above.

| Case | Energy error (Eh) | Force error, 1e-3 Bohr (Eh/Bohr) | Force error, 3e-4 Bohr (Eh/Bohr) |
| --- | ---: | ---: | ---: |
| H2 | 5.00533e-12 | 7.66302e-8 | 2.16240e-8 |
| H3 | 2.99054e-11 | 2.36880e-8 | 2.32613e-9 |
| H4 | 2.66325e-11 | 6.99610e-8 | 2.79390e-7 |

Acceptance remains 3e-8 Eh and 5e-7 Eh/Bohr; no gate was relaxed. The streaming
auditor passes trace/repeat parity, finite forces and the 2e-8 force-translation
gate. Repeat energies are identical; maximum force differences are below
4.4e-16 Eh/Bohr. Timing requests and basis/input snapshots match the contract
described above, including full DF-HF, `standard`, `preconvergence="auto"`, a
32 GiB numeric budget and SCF ceiling 200.

| Case | Setup + first, trace off (s) | Warm median, trace off (s) | Warm median, trace on (s) | Sampled device peak (MiB) | Maximum planned numeric bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| H2 triplet | 1.37373 | 0.18332 | 0.29506 | 6171 | 11,594,937,039 |
| H3 doublet | 20.48090 | 19.03769 | 23.46868 | 6207 | 11,864,888,440 |
| H4 triplet | 320.71717 | 321.71369 | not measured | 6263 | 12,486,832,661 |

Setup excludes imports and compilation. Memory remains 200 ms whole-device
sampling, not allocator-exact peak or process RSS; planned numeric capacity is
a distinct conservative admission quantity. The separate CPU oracle starts
after the trace-off population and can overlap trace-on calls. These populations
do not isolate tracing overhead or establish a hardware/implementation speedup.

| Case | Primal CC solver | Primal (T) | Lambda | Triples response | Orbital response | Nuclear response |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H2 | 0.01251 | 0.000003 | 0.03548 | 0.000011 | 0.00340 | 0.00556 |
| H3 | 0.07256 | 1.04373 | 0.79031 | 20.98009 | 0.02503 | 0.01691 |
| H4 | 0.11214 | 15.15166 | 1.78842 | 401.80824 | 0.02115 | 0.04019 |

Phase seconds use the same per-endpoint normalization and nested-scope caveats
as the historical table. The H4 traced complete endpoint is 421.09588 s; its
three virtual-axis `moment_aaa` programs still execute 1,193,859 actions each.
Ordered triples/moment response remains expensive and finitely bounded, not an
all-size scalability success. Final traces again observe 8/14/26 factor-block
reuses, zero provider AO scans/values after factor preparation,
174,016/1,107,774/4,807,772 initial-weight transform FMAs,
2,304/6,224/15,136 H2D bytes and 23,712/100,832/327,960 D2H bytes across the two
cardinal components. These are semantic UMP2 initial-weight counters, not total
endpoint FLOPs or transfers. No controlled L20 speedup, RCC/UCC ratio or
hundreds-of-AO qualification is inferred.

The [current compact accepted record](../../../../benchmarks/results/unrestricted-df-20261104/qualification-summary.json.xz)
retains these final-source populations and hashes of their full summary and
provenance. Complete inputs, scripts, endpoint receipts, independent oracles,
source inventories, actual library queries, cache/compiler/build logs, device
samples and losslessly compressed traces remain in the ignored
`build/issues-2978-2980-integration-evidence/` directory and source-matched n1
`evidence-integrated-policy` directory. The old `factor-evidence` population is
retained as historical evidence, not substituted for this integration run.
Fresh PR CI is separate from these source-matched local/device qualifications;
no final merged-source GitHub CI pass is claimed here.

## Revisit when

An independently qualified tiled UCC triples/Lambda schedule replaces ordered
evaluation, or a backend supplies bounded device-resident DF nuclear derivative
contractions that remove detached coordinate staging. Preserve complete endpoint
qualification and both metric/orbital response when changing these boundaries.
