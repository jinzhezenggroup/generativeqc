# Decision: default bounded parallel KS diagnostics

Status: implemented in the qualified patch; submission does not requalify later master
Date: 2026-10-06

## Problem

Issue #2001 requests a current-master implementation of the compensated CUDA
KS diagnostic reduction retained in #1965. The historical composed branch is
not the production baseline: this campaign starts at freshly fetched master
`a85459c13d16741ef5fab37cca6a64fc4a5bca15` for both scientific variants.

The final decision is the contiguous second candidate below, not the first
strided candidate. Launch the fixed cooperative owner unconditionally through
the existing production API, without an opt-in. The complete qualification
reproduces the diagnostic bottleneck and a 0.553748/0.448885-second 96-atom
warm/moved-warm endpoint benefit. All eight pooled atom-size/phase medians
improve in this campaign; the full samples and varying cold/moved histories
remain part of the acceptance evidence, not discarded exceptions.

## First mechanism: retained strided candidate

Reuse only the diagnostic owner and its tests from retained commit
`81c9bc783be097a11388e4683032e50f989e057e`, without bringing along the historical
force, Becke, or AO experiments. The public diagnostic API and state arenas
remain unchanged. One fixed 256-thread CTA scans the full spin-resolved matrix
with strided lanes and bounded 26,624-byte shared storage. Each energy trace
retains its Neumaier sum and correction; a fixed ascending lane merge consumes
both words with the existing explicit rounded-add helper. There are no floating
atomics, extra launches, allocations, new scientific owners, method selectors,
or molecule/AO-count/GPU-specific dispatch.

The inactive mask is immutable for a launch and exits uniformly before the
barrier. Every active lane publishes its full partial before lane zero reads
the shared array. Spin-resolved electron counts, squared residual/density norms,
maximum residual, all upstream error bits, solver failures and finite checks
retain the original meaning and strict acceptance thresholds.

This changes summation order, not precision or physical convergence policy.
Bitwise equality to the historical serial order is not assumed. Independent
rounded-product/analytic oracles, actual RKS/UKS/hybrid consumers, sanitizer
coverage and complete endpoints are necessary before acceptance.

## Qualification and promotion

The candidate invokes the parallel owner directly; no opt-in is introduced.
Promotion was conditional on terminal current-head qualification and meaningful
complete-endpoint benefit without regime regressions. The historical roughly
0.45–0.50-second 96-atom warm improvement is a hypothesis, not a gate or a result
for this current-master patch. If qualification fails, retain that evidence
instead of representing an unqualified default as shipping behavior.

Use ccache for the complete CUDA builds of both candidate and serial control.
All real GPU execution runs on n1 via finite `main` / `gpu:5090:1` srun steps,
without modifying scheduler visibility. Exact-direct PBE0 48/96 protocols each
retain cold, five warm, moved and five moved-warm calls, independent numerical
gates and actual histories. Run complete protocols in control/candidate/
candidate/control order within each allocation. Do not normalize by iteration
count, remove outliers, or mix intrusive microbench/profile timing into the
clean endpoint results.

Artifacts and terminal receipts live in the ignored `.artifacts/` directory of
`/data/jzzeng/qc-2001-parallel-ks-diagnostics-20261006`. The original user checkout
is unchanged. At qualification close, no commit, branch, release or distribution
publication had been created by the qualification work.

## Retained first-candidate terminal results

The strided candidate is frozen on n1 at
`/data/jzzeng/qc-2001-parallel-ks-diagnostics-20261006`; subsequent local edits do
not modify those binaries or its benchmark source. Jobs 6164/6163 finish the
48/96-atom control/candidate/candidate/control campaigns: all 96 independent
endpoint gates pass, retaining every sample and actual history. All forty
warm/moved-warm calls per atom-size campaign perform one iteration/Fock build.

Pooled medians are descriptive summaries, not interleaved per-call confidence
intervals. At 48 atoms, control/candidate medians in seconds are cold
125.857993/119.303543, warm 11.453998/11.408002, moved 54.662225/61.635520 and
moved-warm 11.490307/11.286694. At 96 atoms they are cold
275.384630/242.935628, warm 39.209027/38.581820, moved 133.024004/126.746367 and
moved-warm 39.077135/38.645129. No timing is normalized by iteration count.

The 48-atom moved regression is retained: both controls perform twelve Fock
builds, while the candidates perform fourteen/fifteen. The recorded density and
physical-residual metrics are already small near the final steps, but candidate
energy-change values remain just above the unchanged 1e-12 gate. For example,
the first candidate records 1.5916157281026244e-12 at step twelve and
1.3642420526593924e-12 at step thirteen. Do not declare this regime successful
from the 96-atom warm improvement, and do not infer the entire cause without
same-input attribution; upstream matrix/energy accumulation is also part of
the complete trajectory.

Job 6166 profiles exactly one public original-geometry warm call per variant.
The actual diagnostic kernel runs once in each capture, taking
0.478962324/0.004018558 seconds for control/candidate. The isolated preallocated
768-AO RKS scan in job 6162 takes 468.250198/4.020336 milliseconds. These are
mechanism evidence, not clean complete-endpoint timing.

Normal diagnostic tests and memcheck/racecheck/initcheck/synccheck all pass
55 tests each in job 6156, with zero reported errors/hazards. Native job 6161
passes all C++ KS, Fock and XC routes and sixteen independent public hybrid
force tests, but exits one at the subsequent six semilocal tests because their
packaged stationary AOT artifacts were missing. This is retained as a build
prerequisite failure, not a numerical rejection. After building all six AOT
profiles with ccache, the independent packaged semilocal qualification passes
all six cases in a separate frozen checkout in job 6182. PBE0 uses the composite force
owner, not the semilocal packaged branch in `batch.py`; that supplemental
checkout does not change the clean PBE0 artifact inventory mid-campaign.

## Bounded second candidate: contiguous partitions

The first candidate does not establish all-regime acceptance. Use balanced
contiguous partitions with ascending lane merge to retain each spin matrix's
source order within and across chunks, rather than interleave rows by lane.
Quotient/remainder bounds cover every element exactly once without overflowing
`matrix * lane`. This retains the same bounded shared storage, public API,
compensated helper, finite/failure publication and strict thresholds.

This changes memory-access behavior and does not guarantee bitwise serial
equivalence or cure SCF trajectory variation. It must earn fresh numerical,
sanitizer, real-consumer and complete-endpoint qualification. The frozen n1
second-candidate checkout is
`/data/jzzeng/qc-2001-parallel-ks-contiguous-20261006`. Preserve the first
candidate's negative moved evidence regardless of the second result; do not
choose a best repeat or silently replace the old campaign.

## Final second-candidate qualification

Jobs 6187/6188 finish the 48/96-atom complete campaigns with exit zero. Each
allocation retains four complete protocols in control/candidate/candidate/
control order, with twelve independent reference gates per protocol. All
96 endpoint gates pass. Every warm and moved-warm call, including controls,
performs exactly one actual iteration/Fock build. The control is the unchanged
serial implementation at the same pinned master commit, not the historical
composed branch. The offline basis is
`benchmarks/results/pbe0-def2-svp-20261003/def2-svp-ho.json`.

Pooled descriptive medians in seconds, without iteration normalization:

| Atoms | Phase | Serial control | Contiguous default | Control minus default |
| --- | --- | ---: | ---: | ---: |
| 48 | Cold | 126.393818 | 120.612544 | 5.781274 |
| 48 | Warm | 11.192089 | 11.050797 | 0.141292 |
| 48 | Moved | 53.349930 | 52.045826 | 1.304105 |
| 48 | Moved-warm | 11.292908 | 11.139007 | 0.153900 |
| 96 | Cold | 270.803567 | 252.699214 | 18.104353 |
| 96 | Warm | 40.234568 | 39.680819 | 0.553748 |
| 96 | Moved | 140.775997 | 137.199394 | 3.576603 |
| 96 | Moved-warm | 40.188169 | 39.739284 | 0.448885 |

Per-protocol results retain the trajectory variation. Parentheses are actual
Fock-build counts; warm/moved-warm columns summarize their five samples each,
all with one build:

| Atoms | Arm | Cold seconds (builds) | Warm median | Moved seconds (builds) | Moved-warm median |
| --- | --- | ---: | ---: | ---: | ---: |
| 48 | 1 control | 129.792308 (27) | 11.135868 | 53.425479 (12) | 11.288215 |
| 48 | 2 contiguous | 121.385932 (25) | 11.017586 | 51.832409 (12) | 11.043696 |
| 48 | 3 contiguous | 119.839155 (25) | 11.193759 | 52.259243 (12) | 11.154063 |
| 48 | 4 control | 122.995328 (25) | 11.272809 | 53.274382 (12) | 11.298773 |
| 96 | 1 control | 285.788874 (29) | 40.120041 | 148.028692 (14) | 40.149947 |
| 96 | 2 contiguous | 242.265095 (25) | 39.783720 | 147.704089 (15) | 39.855346 |
| 96 | 3 contiguous | 263.133333 (28) | 39.617047 | 126.694699 (12) | 39.641190 |
| 96 | 4 control | 255.818261 (25) | 40.355290 | 133.523302 (12) | 40.269843 |

These are two full protocols per variant/size, not randomized per-call trials
or a statistical confidence interval. Cold/moved histories differ, including
the 96-atom candidate with fifteen moved builds versus fourteen in the preceding
control. Not every individual call is faster: the third 96-atom candidate cold
call is slower than the following control and takes twenty-eight versus
twenty-five builds. Do not attribute the cold/moved median gains solely to this
kernel or claim invariant SCF iteration counts. Unlike the rejected strided
trial, all four 48-atom moved calls now take twelve builds, with both candidate
endpoints faster than both controls. Warm results establish the fixed-work
complete-endpoint benefit without relaxing the convergence policy.

Strict SCF energy/density gates remain 1e-12/1e-10, and independent endpoint
energy/force gates remain 1e-8/1e-7. Maximum absolute energy/force errors are
8.412825991399586e-12/2.5140958759273246e-11 at 48 atoms and
1.064108801074326e-10/3.0266969486270057e-11 at 96 atoms. References are separate
qualification oracles, not production dependencies.

Job 6185 passes all 55 diagnostic tests without skips, both normally and under
each of memcheck, racecheck, initcheck and synccheck. Each sanitizer exits zero;
racecheck reports zero hazards/errors/warnings and the others zero errors.
The tests cover independent rounded-product trace oracles, incomplete/idle
lanes, both spins, full/range exchange, physical metrics, nonfinite inputs,
failure-bit unions, the inactive mask and the output canary.

Job 6186 exits zero for native KS physical-state/warm/failure/resource tests,
independent J/K/direct-through-f/full/SR/LR tests, and all five XC routes
(default, matrix schedule, potential lowering, local AO and PBE0 local AO).
Sixteen public independent hybrid force tests and six public independent
semilocal force tests also pass without skips. Both variants receive full
release CUDA builds using CUDA 12.9.1 and explicit CXX/CUDA ccache launchers:
475 compiler commands per full inventory, zero missing ccache. The six
stationary AOT profiles are built with ccache. Host structure/regression tests
pass 149 cases with nine expected GPU skips outside an allocation. SCF/compiler
audits check 211/465 modules with zero errors; CUDA ownership checks 330 files;
default inventory checks 34 entries and 75 registered controls. Direct Ruff,
clang-format and diff whitespace checks pass; the full pre-commit executable
is unavailable in this environment.

Job 6189 captures exactly one original-geometry public warm call per variant.
SQLite extraction confirms one actual diagnostic launch: control grid/block
1/1 with no shared memory, default grid/block 1/256 with 26,624 shared bytes.
Their diagnostic intervals are 0.478744746/0.004395374 seconds. Job 6192 retains
all six isolated samples per case; RKS medians at 384/768 AOs are
117.048946/1.277840 and 468.061707/4.023104 milliseconds, and UKS medians are
233.990669/2.236544 and 935.498352/7.710896 milliseconds. These intrusive/isolated
numbers explain the mechanism; they are not the clean complete endpoint.

## Evidence identity and limits

Local ignored `.artifacts/qualification/v2/paired-48-6187/` and
`.artifacts/qualification/v2/paired-96-6188/` retain every endpoint sample,
independent reference, actual SCF history, source/binary receipts, Slurm
allocation/step metadata and terminal exit. The consolidated
`.artifacts/qualification/qualification-record.json` includes all 96 raw
accepted endpoint records. `v1-48-summary.json`, `v1-96-summary.json`, their raw
campaigns and `.artifacts/v1-strided/` retain the first candidate's negative
evidence. Production launch extraction, sanitizer XML/logs, native receipts,
full compiler commands and ccache statistics are retained alongside them.

Qualified CUDA source SHA256:
`1b07f3093d2d94e2eed861244b0c4aff7f79e6c8acd0a57c2382246b925e4765`.
Qualified library SHA256:
`6d8b16a42a4c7641d3b7b7cd1b8bf6d6b094d7e76aea7de7cbe3fb74019d9512`.
The frozen source receipt SHA256 is
`bee24cd1de69f20b0f657162c1a238b924d1af92939912d7523a006ea684d24b`.
Its file is `.artifacts/contiguous-drivers/source.sha256`; the complete evidence
manifest is `.artifacts/qualification/evidence.sha256` and can be checked with
`sha256sum --quiet -c .artifacts/qualification/evidence.sha256` from the local
implementation worktree.

On 2026-10-06, master advanced during the campaign to
`05c0bbafb376d60537af2c2c1bfd52ed6469effa`. No production diagnostic source change
is present in those intervening commits, but this qualification is explicitly
at `a85459c13d16741ef5fab37cca6a64fc4a5bca15` plus this patch. Do not relabel its
numbers as a test of the later master or as a merged/shipping performance
baseline. The default is implemented and qualified in the patch. At
qualification close it was uncommitted and unmerged; subsequent patch
submission does not change the pinned test baseline or qualify a later master.

Submission began on 2026-10-06 with master observed at
`ec07c574f806d0c5092a941a59ff9dedba491001`, including the separate force active-AO
workload-cost change in #2007. Keep this patch based on the qualified commit;
do not present its endpoint evidence as qualification of that newer composition.

## Consequences and revisit conditions

The fixed CTA needs no optional storage, additional launches or allocation, so
no resource-dependent serial production fallback is required for this owner.
There is no retained normal-path one-thread scan or feature opt-in. Fixed lane
partitioning/merge order and compensated rounded additions remain invariants;
they do not promise bitwise equivalence to the serial trajectory.

Revisit for reproducible complete-endpoint regressions, diagnostic oracle or
failure-publication failures, new supported-device resource constraints, or a
new qualified summation owner. Any different partition/merge must repeat the
strict numerical, sanitizer, real-consumer and full 48/96 endpoint protocol;
the rejected strided 48-atom moved result is a reason not to accept only warm
microbenchmark wins.

## References and reproduction

Issues #2001, #1965 and #1895 describe the task, retained mechanism and parent
roadmap. Frozen GPU checkout:
`/data/jzzeng/qc-2001-parallel-ks-contiguous-20261006` on n1; serial control:
`/data/jzzeng/qc-2001-parallel-ks-control-20261006`. These are local evidence
artifacts, not publicly published releases or replacement production commits.
With the retained environments/builds, a full size protocol is reproducible as:

```bash
srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --time=02:30:00 bash \
  /data/jzzeng/qc-2001-parallel-ks-contiguous-20261006/.artifacts/run-paired-endpoints-n1.sh 96
```

Run this on n1; use 48 for the other required size. The driver preserves
Slurm visibility, records source/build receipts, produces the independent
reference, and runs all four complete protocols. The sibling
`qualify-diagnostics-n1.sh`, `run-native-n1.sh`, and `run-profile-n1.sh` retain
the diagnostic/sanitizer, real-consumer and actual-launch procedures.
