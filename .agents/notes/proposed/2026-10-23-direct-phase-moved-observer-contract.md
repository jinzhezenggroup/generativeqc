# Diagnostic: Direct component observer acceptance

Status: proposed
Date: 2026-10-23

## Problem

The initial #2678 harness placed the complete `KsDiagnostic` numerical history
under `work` and required exact equality of that dictionary, total energy and
every returned force component between separate clean/instrumented processes.
This agent-added predicate fails between two uninstrumented cold runs as well.
It is not the repository's independently referenced observable acceptance rule.

## Evidence

The bounded cohort is Standard Direct PBE0 and stock GPU4PySCF, 48 atoms
(16 O/32 H), spherical def2-SVP, neutral singlet. Production Python/native/AOT
remain frozen at `ececb48cd56a00a56fbb6223dc53bc61dfe6e11d`; no numerical or
runtime implementation changed. All endpoint diagnostics use complete moving
grid E+F and independently computed CPU references at the exact target geometry.

Job 13586 retained the original bitwise FAIL and stopped before formal timing.
Jobs 13589 and 13594 then used separately owned real GPU allocations for
predeclared clean-A/clean-B/instrumented cold diagnostics. Within each triad,
all processes used the same physical GPU, source, controls and explicit trace
policy. The triad protocol and original raw receipts remain distinct from any
subsequent formal population.

- Native: clean-clean force max/RMS differences were about
  `1.6165e-13`/`4.5306e-14` Eh/Bohr, with equal returned energy. Trace-control
  force max/RMS differences were about `2.0783e-13`/`4.19e-14` Eh/Bohr, with
  total-energy difference `9.094947e-13` Eh. Settings and the predeclared
  available semantic fields were equal: target 9 iterations/9 Fock builds;
  reported Huckel preliminary 7 iterations/7 Fock builds; one target attempt,
  no discarded target work. These are different scopes, not 16 target builds.
- GPU4PySCF: clean-clean and clean-instrumented energy differences were zero;
  force max differences were about `9.5035e-14` and `1.3678e-13` Eh/Bohr.
  All three cases reported 9 iterations and 10 effective-potential invocations.
  Effective-potential invocations are not renamed native Fock/primitive counts.
- Each complete cold endpoint passed the original independent E/F gates.
  Both displaced geometries also have complete CPU and GPU level-4/5 references
  passing the original quarter-budget grid stability and CPU/GPU agreement.

These observations reject a universal bitwise repeatability assumption. They do
not identify the cause of every trace-control difference or prove complete
primitive-work equality. A trace result must not be classified as pure kernel
time, and overlapping API/CUDA-event intervals must not be added.

## Source and counter boundaries

`python/generativeqc/ks_diagnostics.py` defines physical energy/electron/residual
history separately from its integer/domain/branch descriptors. The latter can
be compared exactly without misclassifying physical floating-point diagnostics
as work counters. Keep the full original numerical history in the receipts.

`python/generativeqc/initial_guess.py::read_initial_guess_diagnostic` exposes
guess kind/outcome, reported work and capacity, and preparation wall time. It
does not expose an initial-density matrix identity. Stock guess policy and
public diagnostic agreement do not establish bytewise D0 identity.

The CUDA observer is the existing `TraceOperation`/`TraceRegion` implementation
in `src/runtime/cuda_component_trace.*`; `trace_call` invokes the original
function once. It adds event allocation/recording, synchronization and logging.
The other environment-controlled branches in `src/scf/cuda/df_shell_launch.cuh`
collect profiling resources/counters around the same launch descriptors. The
observed Direct trace does not contain DF shell operations. The Python
`ApiObserver` passes original argument objects, return objects and exceptions;
host tests exercise those boundaries and restoration after failure.

The public Direct Coulomb/exchange census reports `complete=False` and null
application counts. Full primitive work and actual D0 bytes remain
`UNAVAILABLE`; neither is inferred from equal Fock counts. Logical capacities,
constructed grid sizes, actual operator counts and physical floating-point
results must keep their own meanings.

## Proposed acceptance separation

Preserve the original bitwise predicate as a reported diagnostic, including its
FAIL outcomes. Independently require exact scientific settings and documented
available counter/branch agreement, each sample's unchanged independent E/F
gates, and numerical observer stability under the campaign's already declared
quarter observable budget. That budget predates these measurements; do not
choose a new rtol/atol to fit the observed differences. Missing counter coverage
must remain visible and cannot become a claim of all-primitive neutrality.

This proposal does not itself authorize or qualify a formal 28-sample matrix.
The observer contract and its supported scope still require a concrete audit;
the complete cohort, independent review, required CI and protected delivery
remain unfinished. WB97M-V and 96-atom coverage remain part of open #2678.

## Subsequent scoped audit and incomplete collection

On 2026-10-24 the coordinator independently read the hook/decoder sources,
recomputed the original triad vectors and displaced reference gates, and reran
35 host tests. `admit_observer_pair` implements the scoped read-only admission:
required provenance, settings, documented available counters, history branches
and native precision-work inventory must be present and consistent. Missing
fields on both sides are rejected. All original bitwise FAIL diagnostics remain
unchanged. The precision ledger supplies additional reported operator counts;
its completeness flag does not establish primitive recurrence-work completeness.

Cache matching is a separate AND condition. GPU4PySCF's first triad clean run
expanded its private cache from 0 to 418 files, so its startup clock is excluded
from performance/work-comparator qualification. The cache-matched second clean
run and instrumented run both retained the same 418-file census.

Formal Job 13615 collected 14 Native and 8 GPU4PySCF timed samples on one
physical RTX 5090. Native admission, cache and independent observable gates
passed, including both moved geometries. The GPU4PySCF pair passed its numerical
and public-semantic checks, but its clean population naturally added a CUDA
driver-cache file and changed the cache index (421 to 422 files); the
instrumented population stayed at 422. The combined cache gate failed and the
controller stopped before the six GPU4PySCF moved samples. Those eight
cache-unqualified timing samples remain recorded, not ranked or erased.

A targeted recovery was prepared but not submitted in the original window.
Another owner's valid allocation held the required physical GPU until its
natural completion. The release was confirmed with insufficient time remaining
before the original 00:17:01 CST deadline. The collection therefore remains
`INCOMPLETE`: qualified Native 14, cache-unqualified GPU4PySCF 8, and GPU4PySCF
moved 6 not run. This is not a completed cohort or issue delivery. Any future
recovery needs fresh resource evidence and a separately explicit finite plan;
it must preserve the failed attempt and avoid repeating the qualified Native
population or the independent references.

A subsequently approved finite window (00:23:11--00:43:11 CST) queued the
single targeted recovery as Job 13624, `afterany:13620`, with a five-minute
TimeLimit and no requeue. The dependency did not resolve in time. Slurm ended
13624 as `DEADLINE` / `DeadLine`, exit `0:1`, runtime zero and no allocated
TRES. No GPU preflight, E+F or replacement timing sample started. The terminal
StartTime/EndTime values are scheduler transition timestamps, not evidence of
execution. No other owner's job was modified or cancelled. The recovery
opportunity was used by that submission; further retries are not authorized by
that plan. The same incomplete matrix and all prior FAIL records remain.

## Explicit later recovery and bounded offline diagnosis

A new, separately authorized 03:34--03:54 CST window submitted Job 13667 once.
It received the original physical GPU, passed the full identity/occupancy
preflight, and ran only the affected GPU4PySCF clean/instrumented populations.
All eight complete E+F samples passed original CPU observable gates and scoped
semantic/numeric admission. The cache AND condition failed again: clean
expanded 422 to 423 files and changed the index, while instrumented stayed at
423. The controller stopped before moved samples. Slurm recorded FAILED/1:0,
runtime 55 seconds, no requeue/restart. These additional eight timing attempts
are also cache-unqualified; they do not erase or replace the prior failed pair.

Read-only byte inspection then found a concrete harness/layout issue. The two
new CUDA-cache blobs are each 2,380 bytes and differ at exactly one byte: the
digit in the embedded current working directory, `gpu/formal1` versus
`gpu/formal3`. Both contain `default_program` and an empty SM120 PTX program's
version/target/address declarations without a kernel body. The frozen CuPy
13.6 `cupy/cuda/compiler.py` calls `_preprocess('', ...)` while computing its
compiler-version cache key; the default NVRTC program name is `default_program`.
The source file hash is
`9b7bef9a1c7ddc97b9da2ae2cf1f76616e753ad4538aad00b7a22d87cb2e8641`.

This supports CWD-sensitive empty-program preprocessing as the source of these
specific new entries. It does not establish that arbitrary future cache growth
is harmless or prove all primitive work unchanged. No cache blob/index was
normalized, removed, edited or waived, and both original cache FAIL results
remain. A future corrected layout would keep an explicitly fixed working
directory while using independent output/lease directories, subject to a new
explicit execution decision; this diagnosis itself grants no further retry.
The original 188-file local partial archive is preserved separately.

## Fixed-CWD recovery and independently replayed bounded cohort

An explicitly authorized 03:50--04:10 CST window submitted Job 13673 once.
Its shell and each scientific child used the existing `gpu/formal3` working
directory, while all new receipts, logs, temporary files and lease state used
the independent `gpu/fixedcwd1` output root. An executable shared layout guard
checked the actual child CWD, output containment and protected old trees.
Before/after byte censuses confirmed the three original output trees unchanged.
No cache normalization, deletion, counter waiver or hidden priming run occurred.

Slurm recorded COMPLETED/0:0, 04:00:38--04:02:52 CST, runtime 134 seconds,
five-minute limit and no requeue/restart. Full source/environment/SDK preflight
and the original physical RTX5090 UUID passed. The new GPU4PySCF clean and
instrumented populations each kept all 423 cache entries byte-identical; both
moved populations also kept the complete cache census unchanged. This supports
the bounded CWD repair for the observed empty-program metadata defect. It does
not make either earlier cache-changing attempt qualified.

The selected cohort comprises the qualified Native 14 timed endpoints from
13615 and GPU4PySCF 14 from 13673, with four additional cold setups. Independent
local replays by both the slice owner and coordinator recomputed the original
CPU E+F limits, adjacent dense GPU grid stability, CPU/GPU quarter margins,
authenticated public-semantic observer admission, and full within-engine cache
conjunction. All selected endpoints passed. Native and GPU4PySCF began with
different cache inventories (421 and 423); no cross-engine cache equality is
claimed. These are two allocations on the same physical GPU. Original Native
CPU affinity was not captured and remains UNAVAILABLE; the later GPU4PySCF
affinity observations and enclosing process clocks are retained.

All 16 GPU4PySCF timed samples rejected by the two cache gates, original bitwise
FAIL diagnostics, extra cold setups, reference calculations and failed/expired
submissions remain separate evidence. Public-observable admission does not
establish all primitive work neutrality or actual D0 byte identity: both remain
UNAVAILABLE. Complete E+F timings are descriptive; inclusive host API intervals
and inclusive CUDA-event regions are separate, nested and non-additive. Pure
kernel time, isolated J inferred by subtraction, causal cross-engine speedup,
WB97M-V and 96-atom qualification remain outside this slice. Parent #2678 stays
open. Scientific collection and host replay do not constitute PR delivery,
independent latest-head approval or protected merge.

During publication preparation, the owner found that the plan-only CLI path
had not applied the repository's existing raw-output guard at argument parsing.
A new negative regression first reproduced geometry access before rejection;
the bounded fix imports `raw_output_path` in `main` and uses it as the output
argument type. The final host suite has 36 passing tests. This post-measurement
storage-boundary fix changes no endpoint, observation or admission function.
The executed 15f1ed3 driver remains in the reconstruction patch with its original
identity; the submitted driver must not be relabelled as the measured snapshot.
