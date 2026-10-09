# Decision: separate stationary point evaluation from cooperative AO geometry

Status: implemented
Date: 2026-10-09

## Problem

The complete 96-atom PBE0/def2-SVP endpoint after #2160 remains dominated by
integral derivatives and semilocal geometry. A single oracle-checked warm
Nsight Systems capture takes 17.862459 s, with 5.669443 s in integral response
and 5.856605 s in semilocal response. Kernel totals include 4.215855 s in the
generic integral-force complement, 2.146957 s in cooperative geometry (4,608
launches), 1.320083 s in forward phased Becke pairs, and 1.071832 s in the
one-electron shell-warp gradient. These are intrusive diagnostic totals, not
promotion timings, and are not added to host wall regions.

Cooperative geometry combines a thread-zero XC point evaluation, parallel AO
pullbacks, and an inline Becke branch that is never taken for phased owners.
The PBE0 kernel has 255 registers/thread and 72 local bytes/thread. Its unchanged
128-thread block reserves those resources even while only thread zero evaluates
the point model. Reducing AO-label scans alone does not address that frame.

## Decision

1. Reduce the validated AO panel in one ordered traversal. Keep the moving-grid
   accumulator and each consecutive same-atom run private, flushing/reloading
   at run boundaries. Repeated, unordered and noncontiguous maps retain the
   original per-atom and grid addition order. No floating-point atomics are used.
2. For admitted phased, one-point-per-lane geometry, evaluate the existing
   `geometry_point_setup` once per point in a bulk, one-thread-per-point kernel.
   This reuses the shared SCF point algebra and the existing external-seed path;
   it does not introduce a second XC implementation or change thresholds.
3. Borrow the unused inline scratch: store `StationaryPointValue` in its first
   atom channel and external grid-motion seeds in the disjoint grid channel.
   Admission requires `sizeof(StationaryPointValue) <= 3 * na * sizeof(double)`.
   No new retained allocation or host staging is introduced.
4. Instantiate a precomputed cooperative consumer that discards both point
   evaluation and inline Becke at compile time. The shared AO-panel admission
   and scalar-AO fallback are unchanged; the latter also consumes the prepared
   point value rather than evaluating it again.
5. Preserve incoming CUDA launch status and check both new stage launches with
   nonclearing peeks. A failed producer must not admit a consumer of dirty
   scratch, and a failed AO consumer must not admit phased Becke publication.
   Existing borrowed-stream drains and sticky device-error publication remain
   authoritative. No stream synchronization is added.

Compiler code owns the mathematical helpers and emitted kernels. The native
stationary header owns scratch admission and launch ordering. Nonphased owners,
missing optional phased storage and insufficient point-value scratch retain the
original point evaluator. All source channels, AO/Becke selectors, force
screening, precision, point concurrency and resource budgets are unchanged.

## Rejected and superseded variants

- Ordered scalar scatter (v1): the first campaign stopped at an outdated force
  launch-count assertion after some 48-atom samples. It is not qualification.
- Ordered same-atom runs alone (v2): both complete campaigns finish, but fail
  the performance gate. Warm/moved-warm improvements are 0.5975%/0.7345% at
  48 atoms and 0.1884%/0.1518% at 96 atoms. Both arms still have 255 registers,
  72 local bytes and the same theoretical two-CTA limit. Do not report the
  reduced source-level label work as an endpoint win.
- Bulk producer v3: NVCC rejects defaults added to a template after its launch
  references the forward declaration. V4 is withdrawn after an independent
  C++ probe also exposes the remaining function-argument default. Both defaults
  are removed, callers select the template explicitly, and the host test now
  compiles a forward reference before the real definition. Neither variant
  reaches GPU or endpoint qualification.
- V5: completes the independent GPU/sanitizer and both endpoint campaigns. Its
  warm/moved-warm improvements are 7.5899%/6.4619% at 48 atoms and
  4.9598%/5.3168% at 96 atoms. These are predecessor evidence, not relabeled
  measurements of the final binary.
- V6 clarifies the point index name and rebuilds AOT. Its libraries are not
  byte-identical to v5, so v5 timing is not rebound to it. The 48-atom campaign
  completes; the 96-atom campaign is interrupted when v7 adds launch-status
  publication gates. Its partial 96-atom data is not qualification.

Earlier harness failures are retained: a missing compiler `PYTHONPATH`, a
driver indentation failure before any endpoint, a provider-wrapper `nvcc` whose
sibling `ptxas` is absent, and a host declaration regex that did not allow
clang-format's template spacing. Corrections change the harness, not scientific
thresholds. Retain the failures rather than silently overwriting their history.

## Qualification boundary

The benchmark base is frozen #2160 source
`125a4e33f011ed2679d5eedbe298c44e9cefcf98`, not the earlier mixed-worker master.
Both arms use its same production main library SHA-256:
`9563667c7b085c9ed67ac7f175cf27aee9077399584c52e56277005143a801f2`.
Separate, freshly built stationary AOT libraries carry the geometry change.

Final compiler/header source SHA-256 values are:

- `stationary_cuda.py`:
  `0496f7e738cf338af18db8f863c29cb60d99816652d56a41f0c88883340b3155`;
- `stationary_gradient_cuda.cuh`:
  `11b999f68264efae9b269cdfbdfe91f8ed68c98a537b73c6bcdd34463f16031c`.

Use one CUDA context with two independent, immutable native owners. A frozen
baseline compiler/path factory binds its original generator and native header;
the candidate factory binds its own. The pair's input inventory differs only
in those two files. Switching both compiler entry points before owner activity
also covers moved-owner reconstruction. An intrusive observer records the
actual AOT library of the launched cooperative kernel, excluding accidental
symbol/library interposition. Observers pass through during clean timing.

The final patch is based on fetched master
`cdb2131a47aaeb003bbc85fc76cf172848b8044b`, after #2160 merged. Upstream changes
since the benchmark base do not touch the endpoint's native integral, SCF,
DFT, tensor/runtime or Python runtime sources. Three unrelated compiler
GFN2/history files change conservative inventory keys; do not pretend the
whole post-port inventory or main library was benchmarked. Instead, regeneration
proves all four PBE0/B3LYP RKS/UKS SPD wrapper CUDA files byte-identical to frozen
v7, and the generator/native-header hashes above are unchanged. The same
physical CUDA consumer is therefore validated without relabeling the frozen
population. AOT packaging still refreshes its conservative provenance normally.

Each 48/96-atom campaign retains 50 complete calls: two constructions, four
setups, twenty primes, twenty measured calls and four intrusive diagnostics.
There are five interleaved pairs for each warm and moved-warm phase. Timers
include synchronized public energy-plus-force execution and host force return.
All nonsetup calls must have one SCF iteration and one Fock application, no
production fallback, equal scientific work/resource bounds, and equal actual
per-class integral admissions. Both arms launch the same two integral workers.

The retained independent GPU4PySCF references use the original unpruned
48-radial/16-polar/32-azimuth grid, spherical def2-SVP, FP64 and original
acceptance gates: energy <= 1e-8 Eh and maximum force error <= 1e-7 Eh/Bohr.
References are not regenerated or relabeled. Promotion requires improvement
greater than `max(2%, 2 * (baseline relative MAD + candidate relative MAD))`;
this is a descriptive repeatability gate, not a confidence interval.
Independently converged frozen seeds are not asserted byte-identical. Setup and
reconstruction are outside warm timing; no matched cold/moved-first-call or
other-GPU performance claim follows.

## Final v7 results

All 100 complete calls pass the retained independent oracle and work/resource
gates. The four clean comparisons all exceed their 2% repeatability floor:

| Atoms | Phase | Baseline median (s) | Candidate median (s) | Improvement |
| --- | --- | ---: | ---: | ---: |
| 48 | warm | 6.194596 | 5.787014 | 6.5796% |
| 48 | moved-warm | 6.205638 | 5.800456 | 6.5293% |
| 96 | warm | 18.177364 | 17.213449 | 5.3028% |
| 96 | moved-warm | 18.196254 | 17.212110 | 5.4085% |

Maximum energy/force errors are 6.367e-12 Eh / 2.475e-11 Eh/Bohr at 48 atoms,
and 1.001e-11 Eh / 3.790e-11 Eh/Bohr at 96 atoms. Semilocal-response warm
medians change from 1.894395 to 1.472707 s and from 5.929908 to 4.972149 s,
respectively. These components support diagnosis; only complete endpoint
medians establish the speedup. The composed schedule wins, not each change
independently: reduction alone failed above.

Actual cooperative-kernel observations show 255 -> 72 registers/thread and
72 -> 0 local bytes/thread. Blocks remain 128 threads with static shared
storage of 16 bytes. Dynamic shared storage remains 11,648 / 23,936 bytes
at 48 / 96 atoms. Theoretical active-CTA limits change from 2 to 7 / 4;
these are resource limits, not observed occupancy. Diagnostics bind the exact
baseline and candidate AOT paths in both geometries. The final PBE0 RKS SPD
AOT library SHA-256 is
`a0c7f89f8bdbdfdf04f1d34af4a16168693e4a8a4fcd9c46c8dc052be9ba469a`.

Raw 48/96-atom record SHA-256 values are
`e68a719d51725df049a71c789b528b7ce8a13ef9563dfbfaa5e6ff37e200cc5f` and
`da10525e447ac10527eda7bf32f04f698d7c64480e17c6a61c021d412067eb7a`.
Retained reference SHA-256 values are
`dea65338702bca3cea6676c3b0e26b17f767e730fcdcbc4f9f7852a7337f555e` and
`b0961774cc1a4e2a729d205a3952c6d2ec94e6593d5ead98226e3ac982fe8d9e`.

The final post-master-port host cohort passes 358 cases, including exact-bit
ordered reductions, scalar/panel admission, external offsets, empty/repeated
maps, sticky device status and compiled launch-failure injection. GPU v7 passes
360 bounded/phased composition cases and 12 independent Libxc energy-difference
cases, including four 96-atom/1856-AO semilocal-slice fixtures. These do not
extend the public through-f or nonlocal full-force contract. Memcheck,
racecheck and initcheck each pass four selected real-owner cases: zero errors,
and zero race hazards/warnings. Compiler ownership checks 504 modules with zero
errors; current-master source-bound materialization review retains six sites,
with no added, removed, changed or unreviewed sites.

The separate final v7 Nsight Systems capture, on the same assigned GPU after
the clean campaign, takes 17.310212 s with one iteration/one Fock, no warm
fallback, energy error 5.457e-12 Eh and force error 3.770e-11 Eh/Bohr. It is
intrusive diagnosis only. The split point/AO kernels total 0.944797 + 0.282761 s
across 4,608 launches each, versus the earlier combined geometry's 2.146957 s.
The remaining largest kernels are generic integral force (4.358578 s), forward
Becke pairs (1.348521 s), one-electron shell-warp gradient (1.094001 s), and
the bulk point producer itself (0.944797 s). This identifies future targets;
do not treat profiler differences as a matched speedup or add them to endpoint
wall times. Raw reports, SQLite export, CSV and checked complete-call record
remain in the v7 evidence directory.

## Evidence and reproduction

Ignored local evidence lives under `.artifacts/next-hotspot/`; frozen source,
libraries and complete raw records are also under node1
`/data/jzzeng/qc-next-hotspot-20261009-125a4e33f`.

- `evidence/ordered-run-v2/`: rejected reduction-only complete campaigns.
- `evidence/bulk-point-v3/` and `bulk-point-v4/`: rejected build populations.
- `evidence/bulk-point-v5/`: complete precursor qualification.
- `evidence/bulk-point-v6/`: superseded qualification, including partial 96 data.
- `evidence/bulk-point-v7/`: final source/binary identities, raw complete vectors,
  verifiers, real kernel bindings/resources, independent GPU and sanitizer logs.
- `master-source-proof/`: both sets of four emitted wrappers and exact hashes.
- `host-master-final.log`, `structure-master.log`,
  `materialization-master.json`: current-tree host, ownership and source review.

Reproduce builds with `build-next-v7.sh`, numerical gates with `gpu-next-v7.sh`,
complete timing with `paired-next-v7.sh`, and the separate intrusive capture with
`profile-bulk-v7.sh`. All real-GPU work uses finite Slurm `main`/`gpu:5090:1`
allocations and assigned visibility. Compilation reuses ccache. Do not overwrite
source populations or turn interrupted/profiler data into clean timing samples.

## Revisit when

Profile the qualified endpoint before choosing the next target. Integral-force
complement, phased Becke pairs and one-electron response remain independent
candidates. Do not blindly extend order-six cooperative integral force: the
retained negative experiment regresses 1.995 s to 3.480 s. Order-seven and
one-electron changes need their own complete-endpoint and work evidence.
Likewise, a theoretical occupancy limit is not observed occupancy, and lower
shared-memory admission alone does not prove a speedup at fixed point tiling.

## References

- #1893, #1894, #2155, #2160.
- `2026-10-03-cooperative-ao-point-panel.md`.
- `2026-10-06-cooperative-weighted-shell-force.md`.
- `docs/maintainer/performance_engineering.md`.
