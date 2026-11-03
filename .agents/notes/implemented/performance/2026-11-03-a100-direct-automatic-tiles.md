# Decision: guarded automatic A100 fully Direct SCF/force tiling

Status: implemented; scoped default-grid scheduling qualified; broader/identity limits retained
Date: 2026-11-03

## Follow-up: native architecture-independent fallback

The default-policy decision below is **superseded** by PR #2948 after review.
Neither an A100 name check nor an exact SM80 equality is an endpoint-speed
qualification. Fully Direct native automatic SCF and force tile selection now
retains 256 on every GPU unless the caller has an explicit tile or the Python
frontend has selected an independently qualified exact local DFT schedule.
The existing `generativeqc-tune` search already binds accepted profiles to
source/device/workload identity and complete energy/force numerical, resource,
and performance gates. Native C++/C callers can select a measured tile
explicitly; their default path does not load Python's local profile cache.

**Performance tradeoff:** the A100 data below remain valid for the recorded
water-96 population, but no longer trigger a 1024-point default without a
matching qualified profile or explicit configuration. Untuned A100 Direct
endpoints can regress to the old 256-point timing. This is deliberate until
portable, evidence-backed native profile consumption is implemented; it
must not be represented as performance-neutral. The fitted scheduler is
unchanged, and nothing is inferred for H100, A800, RTX 5090 or other devices.

The original evidence, conclusions and failed diagnostics below are retained
as historical provenance rather than edited into unmeasured performance claims.

## Problem

The preceding [Direct orbit investigation](2026-11-02-direct-force-orbit-and-tuning.md)
found that explicit SCF/force tiles of 1024 reduce complete A100 endpoints, but
left fully Direct automatic owners at 256. It also left the restricted-orbit
coefficient path without an A100 production build because CMake selected Python
3.10. This task resumes that deferred work, rather than extending HF autotuning
coverage to DFT or substituting microkernel measurements for complete endpoints.

## Decision

Reuse the existing independent SCF and stationary-force tile owners. Admit 1024
automatically only for device-fused, full-range, gradient-dependent, strict-FP64
RKS with no application arithmetic budget, on SM80 (compute capability 8.0)
devices reporting FP32/FP64 throughput ratio two. Require all-electron shells
through d, 48–128 atoms, at least 512 AOs, 524288 actual XC grid points and
1 GiB free device memory. The SM80/FP64 hardware gate does not infer performance
on H100, RTX 5090 or future architectures from throughput alone.

A review follow-up removed the original `device_name.find("A100")` gate:
driver/SKU/MIG labels are not stable capability identifiers. Other SM80 devices
reporting the same FP64 ratio now pass the hardware predicate, but only A100 has
measured complete energy-and-force endpoints in this note. This change is not
evidence of a performance improvement on any newly admitted device.

Explicit SCF and force bounds retain priority, including explicit SCF-to-force
inheritance. Numeric capacity promises, active resource ledgers and reference
memory budgets retain their conservative scheduling. Unqualified automatic
owners retain 256. Existing preparation and force-controller allocation retries
are still bounded: drain partial borrowers, retry once at 256, and propagate a
second failure or a non-allocation failure. Explicit requests never silently
shrink. Density-fitted scheduling is unchanged.

The strict arithmetic-budget check also applies to Direct SCF admission, not
only force admission. FP64 mode and an application error budget can coexist;
checking the mode alone would export a strict measurement to another contract.

## Invariants

- Preserve FP64, method-defined orbital bases and grids, moving-grid response,
  screening tolerances, source coefficients and SCF stopping rules.
- Do not import PySCF/GPU4PySCF in a native timed endpoint or reuse a reference
  density. Each cold owner prepares its own seed; repeated warm calls freeze
  that owner's cold seed instead of propagating repeat-dependent solutions.
- Keep the single bounded Direct quartet traversal. No CUDA kernel, scientific
  recurrence, register bound or dispatch enumeration is changed here.
- Keep complete construction/preparation/SCF/forces/host-export timing separate
  from intrusive event traces and synchronous force-work census owners.
- Preserve null unavailable work attribution. Equal iterations alone do not
  prove equal Coulomb, exchange, incremental or primitive work.

## Source, build and execution provenance

Use the existing `/data/jzzeng/direct-autotune-20261102` worktree at `874366d99`.
The user's original `/home/jzzeng/codes/qc-gencode` edits are not transferred or
modified. Scientific identities are:

| Image | Scientific source identity |
| --- | --- |
| Previous master | `4b8ab33e04c4347589fdaa232c6eb3a3e8a079c5385d827f0a39dea564a01f51` |
| Frozen guarded-orbit baseline | `13f12658f4d7821c246cf95121b554e28127edc74297a3a72834932ce0bb328f` |
| Initial automatic-tile screen | `617c87679106688f982adb3e98f9db6b2239ea0a3931a51ef0648a912bfe15d4` |
| Final strict-budget candidate | `0d6240a0b19b3d313d884fbc2a476a132555898bceca03e083b9f8224675ee10` |

Frozen baseline main ELF SHA256:
`fba8d262897decd8b2dfb2b87a118a80972abe248d79d61a545e5f461845122a`.
Final candidate main ELF SHA256:
`de094a0fabcee34788555d4a59d12cb693342bdcbcb1e61604a434df6774f8fd`.
Every endpoint wrapper recomputes scientific inventory identity, checks all
packaged binaries and compares the loaded main ELF's exported identity before
scientific work. Each receipt retains actual LSF and CUDA visibility fields.

All remote GPU work uses finite LSF allocations on ECNU `gpu_chem` / `gpu22`,
one exclusive-process A100-PCIE-40GB, preserving assigned visibility. No Slurm
variables are fabricated. The unrelated suspended user job is left untouched.
Production Release images target `80-real`, disable fast compile and test-only
kernel coverage, and retain stationary AOT bindings. Pin Python 3.12 explicitly.
Reuse sccache 0.16.0 with task-owned foreground servers; do not clear the shared
cache or stop other servers. Final build 224956 has four CUDA-family cache hits,
two C++ misses and no compilation/cache failures. Private runtime library paths
are scoped to individual Python invocations, not exported into compiler shells.

## Initial screening evidence

LSF 224955 compares the frozen baseline, initial automatic image, explicit 256,
and (for PBE0/r2SCAN-3c) frozen explicit 1024 and previous-master controls.
All 22 owners complete their cold plus three frozen-density warm E+F calls.
All six methods improve both cold and median warm endpoints in this screen;
the initial source is **not** the final strict-budget image above.
`screen-224955-summary.json` retains all timings and all-pair numerical results.

Across every initial-screen sample pair, native error maxima are 1.90e-14
Eh/atom, 1.54e-11 maximum and 2.87e-12 RMS force error in Eh/Bohr. Against the
retained independent GPU4PySCF populations the maxima are 9.84e-12 Eh/atom,
1.35e-6 maximum and 4.50e-7 RMS force error. Gates remain 1e-10 / 2e-9 / 1e-9
for native parity and 1e-5 / 1e-4 / 3e-5 for default-grid independent parity.
No failed numerical pair is discarded.

The initial frozen/candidate `cuobjdump --dump-resource-usage` outputs are byte
identical. Explicit 256 restores the baseline scheduling; automatic 1024 agrees
with the frozen image's explicit 1024. PBE0 has 9 cold / 1 warm Fock builds;
r2SCAN-3c has 10 / 1. These controls support a scheduling attribution, not a
register/occupancy claim.

## Acceptance and retained evidence

Final-source job 224957 completes the six-method static comparisons, explicit
256 controls, 48-atom controls and all four changed-geometry owners. It then
fails on the **frozen baseline's** 192-atom cold SCF request, before producing an
accepted endpoint. The candidate 192-atom request is not run. Do not substitute
that failed request for a latency sample, imply both images were tested there,
or describe it as a timeout. Stock controls, including the SCF limit, are not
relaxed. The artificial 192-atom fixture is two water-96 clusters translated by
(45,5,3) Bohr, not a continuation of the nested benchmark series.

Job 224962 separately runs independent moved-geometry references, diagnostic
force census/event owners and 32-GiB resource-promise controls, then tests the
automatic boundary at 129 atoms. That separate fixture is a water-96 cluster
plus a translated 33-atom prefix, with an explicitly named provenance tag. It
does not replace or erase the unsuccessful 192-atom experiment. Both 129-atom
owners complete cold and frozen-density warm E+F calls at resolved SCF/force
tiles 256/256, with 10/10 and 1/1 iterations/Fock builds. Maximum paired force
difference is 4.21e-13 Eh/Bohr. Both 32-GiB resource-promise owners also resolve
256/256 and pass native and independent default-grid gates. The independent
large-size accuracy gate is not claimed from a native-only 129-atom comparison.

The two jobs retain 32 complete endpoint cohorts (112 E+F calls) plus four
five-call changed-geometry owners. The merged scope receipt explicitly reports
successful matched-state, independent default-grid, boundary/resource and
bounded-work checks, while keeping exact cross-trajectory restoration and the
broader 192-atom endpoint unqualified. Both jobs use actual assigned visibility
`0`, sequentially, not overlapping GPU allocations.

### Complete default-grid owner timings

These values use the final strict-budget scientific source, not the initial
screen image. Cold includes construction, preparation, SCF, analytic forces and
host export; import/provenance checks are outside the timed public boundary.
PBE0 and r2SCAN-3c each have three fresh cold owners and nine frozen-density warm
calls per arm. Their second round reverses image order; every round's timestamps
confirm sequential nonoverlapping owners. The other four methods have one cold
and three warm calls per arm. All values below are medians in seconds, not
fastest samples or a global optimum.

| Method | Frozen cold | Automatic cold | Frozen warm | Automatic warm | Cold / warm reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| PBE0 | 16.7522 | 14.8611 | 6.4145 | 5.3498 | 11.29% / 16.60% |
| r2SCAN-3c | 21.3501 | 19.1852 | 11.6214 | 10.4221 | 10.14% / 10.32% |
| PBE | 9.9402 | 7.9641 | 4.1117 | 3.0346 | 19.88% / 26.20% |
| B3LYP | 16.9011 | 15.0692 | 6.4881 | 5.3689 | 10.84% / 17.25% |
| r2SCAN | 11.0010 | 8.6730 | 4.5536 | 3.3142 | 21.16% / 27.22% |
| M06-2X | 19.5376 | 17.2788 | 6.7008 | 5.5769 | 11.56% / 16.77% |

Cold iterations/Fock builds are respectively 9/9, 10/10, 10/10, 9/9, 9/9 and
10/10 in this table's order; every warm call has 1/1 in both images. Recorded
Coulomb/exchange applications agree as well: PBE0 has 9/9 cold and 1/1 warm,
while r2SCAN-3c has 10/0 cold and 1/0 warm. Incremental anchor/delta/closure counts
are retained; unavailable SCF quartet counters are not reported as measured zero.

The initial Bash qualification helper accidentally accumulated arm prefixes in
some output filenames because its temporary label was global. Actual image
selection, method arguments and round order remain correct and are verified
against exported source identities and recorded start/completion timestamps.
Analysis retains original filenames, adds an explicit canonical measurement
label and rejects duplicate or missing roster entries. No raw record is renamed
or overwritten. The reusable helper now uses local function variables.

### Semantic work, including amplification

Separate job 224962's cold/warm Direct-force census files are byte-identical
between frozen and automatic images, including all per-class rows, traversals,
screening, admissions, execution counts and null unavailable fields. Both calls
in each owner report complete bounded s/p/d coverage. Actual primitive products
are 695763928 per PBE0 force call and 1444681762 per r2SCAN-3c call. The matching
census file hashes are respectively
`942f068d651d4e66053f7a1866d28d3845bb8d20bc3a6b47bbe6b047dc9b30e5` and
`586a2fe3276a1c928bdf69412c883016fafe4f4aa6df0615f75a54100f470715`.

This does **not** mean all algorithmic work is unchanged. The last warm
`stationary_force` event records the same 1078336 quadrature points but 4213
versus 1054 evaluation tiles. Larger tiles include additional local AOs:

| Method | Active point-AO visits, 256 / 1024 | Projection FMA-pair counter, 256 / 1024 |
| --- | ---: | ---: |
| PBE0 | 224294592 / 248849856 | 207471842560 / 269510873344 |
| r2SCAN-3c | 327106432 / 363634304 | 447559159808 / 578788230144 |

Those are semantic counters, not hardware FLOPs. The measured endpoint improves
despite this local-grid work amplification. This tradeoff is why 1024 is not an
unrestricted device/geometry default. Intrusive CUDA-event and census latencies
are never mixed into the clean timing cohorts, nor summed as nonoverlapping
endpoint components.

### Changed geometry and a retained failed identity diagnostic

Every same-phase baseline/candidate cold, warm, moved, moved-repeat and restored
comparison passes the unchanged strict native parity gates. Maximum matched
force differences are 3.49e-13 Eh/Bohr for PBE0 and 1.19e-11 for r2SCAN-3c.
Fresh fully Direct GPU4PySCF references for the moved geometry, plus retained
original-geometry references, independently accept all twenty native results
at the unchanged default-grid observable gates. Maximum independent force
errors are 1.23e-6 and 2.20e-6 Eh/Bohr respectively; RMS errors stay below
7.78e-7 Eh/Bohr. This is not a grid-converged accuracy ranking.

An additional **cross-trajectory** diagnostic, restored-versus-own-cold at the
much tighter native identity thresholds, does not pass in either image:
PBE0 differs by about 9.02e-7 and r2SCAN-3c by 3.40e-6 Eh/Bohr. These four failed
checks remain explicitly failed in the analysis; no tolerance is increased to
turn them green. The restored solve takes two Fock builds, whereas resident
frozen-density replay takes one. Geometry rebuilding materializes the retained
seed and re-enters the existing host warm-input normalization boundary; freezing
snapshot updates does not guarantee an identical SCF trajectory through that
boundary. This observed pre-existing spread is not evidence of a new tile
regression, but neither is exact restoration identity qualified here. Its deeper
solver/state investigation is outside this scheduling change. Do not report
"all diagnostics passed" or silently discard the failures.

Independent A100 Libcint derivative gates cover RKS PBE/PBE0 and UKS PBE0,
Cartesian/spherical d-bearing fixtures and separate J/K sources. Initial-screen
baseline and candidate each pass six tests; final-source job 224957 repeats and
passes six tests per image. Host force/math/screening tests pass
37 tests; final policy/scheduling/inventory tests pass 37 (including the three
tile-policy tests). Compiler structure checks 583 modules with zero dependency
errors; Ruff, clang-format and `git diff --check` pass.

Raw receipts and helper sources are retained under
`/home/jzzeng/codes/qc-gencode/.artifacts/a100-direct-20261103/`:

- `build-224956.tar.gz`: `58a650022f2786a5aad9cad97e6c0a84fa87c70fb8e1322159ed800b037a17aa`.
- `screen-224955.tar.gz`: `3003ad9c2882519d14106b0789cfe2e97a3eb7b791f64c1b4e952dc744ae0547`.
- `qualify-224957.tar.gz`: `5dacbbaf8b0e376dd49708bcf9f27507fbf75494b970925b790143b8691e51e2`.
- `followup-224962.tar.gz`: `7b0d7d9015e47182ff0933690425ef6312c3ad56df868d8359476e0cba743725`.
- `analyze.py`, `endpoint.py`, `moved.py`, `qualify.sh`, per-image identity
  receipts, full failed/build logs and every numerical/timing sample remain local.

`qualification-summary.json` is the scoped merged receipt. Reproduce analysis
from the retained task artifact directory with:

```bash
python analyze.py followup-224962 --baseline-directory qualify-224957 \
  --output followup-224962-summary.json
python analyze.py qualify-224957 --retain-incomplete \
  --moved-reference-directory followup-224962 --output qualify-224957-summary.json
python merge-qualification.py
```

The second command deliberately exits nonzero for the four retained strict
restoration-identity failures while still publishing its complete analysis.
The merge does not relabel them as passes: its acceptance map keeps both
`cross_trajectory_exact_restoration` and `broader_192_atom_endpoint` false.
`--retain-incomplete` preserves the unsuccessful 192-atom record; it does not
turn it into a completed or timed endpoint. The unchanged strict matched-state
native and independent observable gates remain mandatory for the accepted scope.

All task GPU work has finished. The task-owned temporary transfer server and
unused browser helper are stopped after local receipt verification. Compiler
caches, remote scientific images, raw evidence, the existing VPN session and the
unrelated suspended user job are preserved. At the initial A100-only handoff,
no commit, PR or Release had been created.

## RTX 5090 non-regression follow-up

The subsequent PR request explicitly requires preserving RTX 5090 performance.
Slurm job **15714** completes on authorized `n1` (`node1`, partition `main`) with
one scheduler-assigned RTX 5090, a finite 40-minute limit and exit status 0:0.
Runtime is 30:16. All 36 native owners authenticate the same assigned visibility
`2` and physical PCI bus `0000:41:00.0`; none overlap. No job runs on the
maintenance node. Slurm accounting storage is disabled, so the receipt retains
that failed metadata query as well as the successful final `scontrol` record
showing `COMPLETED`, the finite time limit and successful exit.

Both SM120 production images use Release, tests off, fast compile off and the
same native stationary image layout. Their scientific identities are the frozen
`13f12658f4d7821c246cf95121b554e28127edc74297a3a72834932ce0bb328f`
and final strict-budget
`0d6240a0b19b3d313d884fbc2a476a132555898bceca03e083b9f8224675ee10`.
The loaded main ELF hashes are respectively
`4c16a1970a98c1a0ed80bf3300b7b58517ee16324121801bad5dc13988f651d6`
and `ffc7c5fb6676eee546aa5274ce58460c35ec1be577a8411d1fbbf09f02aa9545`;
the retained wB97M-V bindings/manifests are byte-identical. The existing warm
SM120 build and verified sccache 0.16.0 launcher are reused without clearing
caches; before/after statistics accompany the full build log.

Before execution, `acceptance.json` declares a maximum candidate/baseline median
ratio of 1.05 independently for cold and warm endpoints, unchanged numerical
gates and unchanged semantic work. The six stock fully Direct water-96 methods
retain FP64, bases, default grids, screening and stopping rules. PBE0 and
r2SCAN-3c each have four fresh cold owners and twelve frozen-density warm calls
per image; the other methods have two cold owners and six warm calls per image.
Odd rounds run baseline/candidate and even rounds reverse that order. The 32
clean owners supply 128 complete E+F samples; all samples are retained. Four
additional two-call diagnostic owners are excluded from timing claims.

| Method | Frozen cold / warm (s) | Candidate cold / warm (s) | Candidate time change cold / warm |
| --- | ---: | ---: | ---: |
| PBE0 | 19.1793 / 7.1492 | 19.1001 / 7.1495 | -0.413% / +0.005% |
| r2SCAN-3c | 31.3434 / 14.0224 | 31.3490 / 14.0583 | +0.018% / +0.256% |
| PBE | 13.4886 / 5.1898 | 13.5046 / 5.1898 | +0.119% / +0.001% |
| B3LYP | 19.5475 / 7.3693 | 19.5639 / 7.3729 | +0.084% / +0.049% |
| r2SCAN | 15.6147 / 6.2577 | 15.7108 / 6.2569 | +0.615% / -0.012% |
| M06-2X | 23.2761 / 7.7574 | 23.2618 / 7.7563 | -0.061% / -0.013% |

Every result resolves automatic SCF/force tiles to **256/256**. Cold
iterations/Fock builds remain 9/9, 10/10, 10/10, 9/9, 9/9 and 10/10 respectively;
all warm calls remain 1/1. Coulomb/exchange and incremental anchor/delta/closure
counts match between images. PBE0 and r2SCAN-3c force census files are each
byte-identical, including traversal/screening/class counters and unavailable
null fields; primitive products remain 695763928 and 1444681762 per call. Thus
the result observes no material regression in this measured population, rather
than proving zero overhead or performance equivalence for every workload.

All 768 clean native cold/warm cross-pairs pass the unchanged gates. Maximum
energy/atom, force max and force RMS differences are 2.37e-14 Eh, 1.46e-11 and
2.90e-12 Eh/Bohr. Retained independent GPU4PySCF default-grid references accept
every clean native sample, with maxima 9.85e-12 Eh/atom, 1.35e-6 and 4.50e-7 Eh/Bohr.
Those independent endpoint references were acquired on A100; they are not new
5090 reference timing observations. Fresh 5090 Libcint derivative checks pass
six tests, and fresh focused host policy/transport/inventory/control tests pass
31. Host policy cases explicitly reject the complete RTX 5090 SM120/ratio-64
configuration at 48, 96 and 128 atoms. No production source changes during this
follow-up, so the A100 qualification above remains source-matched. Its retained
restoration-identity and 192-atom failures are not changed or relabelled.

Local evidence is retained under
`/data/jzzeng/direct-autotune-20261102/.artifacts/a100-direct-5090-20261103/`:

- `5090-15714.tar.gz`: `d7bb7dbb33505699315a8531864def5cee57b5b863142c9ebc50a4a50f63c12f`.
- `candidate-source.tar.gz`: `28cbf55fb74ff6e2945969609b4e17435792e8dcd3bef9fbd96f619904fc417e`.
- `candidate-production-sm120.tar.gz`: `0923a7d2c1090afda500770dfb4e6775a2a29e05a9f0e0ad006aa78706757309`.
- `qualification-summary.json` retains every clean/diagnostic cohort, paired
  gate, method median, all latency samples, semantic work and sequential owner
  metadata. The raw archive includes the unchanged predeclared acceptance
  receipt, executable wrappers/analyzer, all observables and scheduler records.

Reproduce from that local artifact directory, without running GPU work:

```bash
python analyze.py 15714 --acceptance acceptance.json \
  --references ../direct-autotune-20261102 --output qualification-summary.json
```

## Rejected alternatives and revisit conditions

The [entry-only register-cap experiment](../../rejected/2026-11-03-direct-entry-only-register-cap.md)
fails compilation because capped entry kernels call 255-register out-of-line
callees. It has no measured GPU endpoint. Earlier angular partitioning amplified
candidate scans and remains rejected. Neither experiment is retained in CUDA.

Do not promote this result to a dense-grid accuracy ranking, all-GPU claim or
general molecular benchmark win. Independent grid-converged references and wider
molecular populations remain separate qualifications. Revisit broader device,
geometry, basis/precision or register-policy admission only with fresh complete
endpoints, independent observables and semantic work/resource evidence.
