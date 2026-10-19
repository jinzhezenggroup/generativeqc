# Decision: admit phased Becke geometry in compiled native LDA/PBE forces

Status: implemented and GPU-qualified within the scope below
Date: 2026-10-19

The profile restriction and qualification snapshot below are superseded by
[functional-independent native admission](2026-10-19-functional-independent-phased-becke.md).
These original LDA/PBE-only measurements retain their historical source identity.

## Problem

The [pruned endpoint investigation](2026-10-19-pruned-endpoint-attribution.md)
identified repeated cooperative geometry AD as a substantial remaining cost.
The compiled native stationary controller configured the cooperative executor
but did not admit the existing phased executor used by Python/composite owners.
Moving AO discovery to the device gave only about 1.006–1.007x complete-endpoint
ratios, so that unrelated default was deliberately not promoted.

## Decision and invariants

- Emit the 48-atom phased crossover only for compiled `lda_rks` and `pbe_rks`.
  Keep the existing 128-atom cap, full-point-lane requirement, retained-center
  prerequisite and actual sm_120 capability checks. Other profiles are unchanged.
- Reuse the existing generated phase equations and executor. No grid, pruning,
  cutoff, precision, SCF stopping rule or integral approximation changes. Mixed
  geometry still uses its existing executor; this is not mixed-precision promotion.
- Charge optional pair panels to the authoritative **cached** capacity before
  complete-owner admission. Do not enlarge the mandatory minimum arena. A cached
  tier miss, unsupported device, reduced lanes or optional allocation miss retains
  bounded geometry. At 96 atoms / 256 points the optional panel is 39,755,392 bytes.
- Configure before topology. Preserve typed setup errors and destroy the scientific
  owner on configuration failure; do not reinterpret driver/identity failures as
  allocation misses. No CPU/reference work enters production.

This extends the retained policy in
[composite phased reservation](2026-10-14-composite-phased-becke-reservation.md),
not a second Becke implementation. The current contract lives in
`docs/developer/stationary_cuda_scheduling.md`.

## Clean complete-endpoint measurements

Both binaries use clean source `6c72988221a827167d57f9156b407125c8c0db9b`;
the candidate adds only the runtime/emitter patch, excluding unrelated dirty
float-float/VV10 work. RTX 5090, direct RKS/STO-3G, default qualified pruned grids,
one CPU/BLAS thread, process/artifact warm. Each sample constructs a fresh
Calculator, prepares, converges SCF, computes analytic forces, destroys the owner,
collects garbage and synchronizes. No cross-sample converged-density reuse.
One complete warmup per binary precedes six alternating samples per method.

| Atoms / method | Baseline median (range), s | Candidate median (range), s | Ratio | Wall reduction |
| --- | ---: | ---: | ---: | ---: |
| 96 / LDA | 21.110314 (21.090758–21.122190) | 14.688548 (14.672043–14.702683) | 1.4372x | 30.42% |
| 96 / PBE | 15.247510 (15.218133–15.267635) | 11.143711 (11.116675–11.156542) | 1.3683x | 26.91% |
| 48 / LDA | 7.477714 (7.388968–7.573523) | 6.745923 (6.630683–6.861526) | 1.1085x | 9.79% |
| 48 / PBE | 6.016001 (5.922509–6.116515) | 5.638225 (5.559342–5.783475) | 1.0670x | 6.28% |

Both engines here are GenerativeQC before/after, not a new GPU4PySCF ranking.
All samples perform 15 reported Fock builds. LDA/PBE point counts remain
2,467,200 / 1,579,008 at 96 atoms and 1,233,600 / 789,504 at 48 atoms. There is
no VV10 or UKS performance claim, and 128 atoms have numerical, not timing,
qualification in this change.

Slurm 10402 reserves four scheduler-visible GPUs and explicitly selects local
device 3 from unchanged visibility `0,1,3,4` (physical GPU 4, verified idle before
execution). Job 10406 uses assigned visibility `5`, local device 0. This avoids
unrelated active workloads without overriding visibility. The initial contended
pilot 10398 was cancelled and its partial results retained, not pooled with this
population. Initial GPU test job 10399 failed during preparation with
`std::bad_alloc` while busy devices were near capacity; it is retained as a failed
attempt, not called a numerical pass. Later idle-device tests pass unchanged.

## Numerical and resource qualification

At 96 atoms, baseline/candidate energy differences are zero in all six samples;
maximum force differences are 1.10134e-13 (LDA) and 1.17240e-13 (PBE) Eh/Bohr.
At 48 atoms, maximum energy/force differences are 6.82121e-13 Eh / 1.55432e-13
Eh/Bohr. The explicit parity gates are 1e-9 for both observables.

Independent dense GPU4PySCF level-9 references and level-8/9 stability margins
are unchanged from jobs 10323/10324. Maximum candidate errors including that
observed margin pass the existing production gates:

| 96-atom method | Energy including margin / gate, Eh | Max force including margin / gate, Eh/Bohr |
| --- | ---: | ---: |
| LDA | 2.127145e-5 / 3.2e-5 | 1.428920e-5 / 5e-5 |
| PBE | 4.311685e-5 / 4.8e-5 | 1.217117e-5 / 3e-5 |

`test_native_stationary_phased_budget.py` executes generated policy, real capacity
arithmetic and native creation with inert allocation/configuration boundaries.
It covers exact/one-byte-short/minimum budgets, reduced lanes, excluded profiles,
typed phase errors and cleanup. The focused and extended host suites pass
119 + 34 tests, with six GPU cases skipped locally. Compiler structure checks
569 modules with zero dependency errors; owned Python/C++ formatting checks pass.

Job 10402 then passes all six allocation-gated CUDA cases: LDA/PBE at 48/96/128
atoms, original and moved geometry, fully reconverged energy differences at steps
2e-4 and 1e-4 with a 1e-7 derivative gate, and fresh minimum-tier force parity
(1e-7 force / 1e-9 energy gates). Separated H2 fragments isolate derivatives;
they are not substituted for an absolute external energy reference. Job 10403
passes memcheck and synccheck on the 48-atom LDA case, each with zero errors.

Nsight job 10405 independently proves route selection: the fresh PBE force
executes all seven phase stages 6,168 times each, plus one index initialization;
the former `geometry_cooperative_kernel<false,double>` is absent. The geometry
consumer now uses its precomputed variant. These are intrusive route/work
observations, not additive clean-wall fractions or separate speed claims.

## Provenance and retained evidence

On n1, root
`/home/jzzeng/generativeqc-qualification/pruning-optimization-20261019/`
retains the clean HEAD archive, separate baseline/candidate binaries, the applied
production patch, compiler commands and sccache-before/after snapshots. CXX and
CUDA launchers use verified sccache 0.16.0; the shared cache was not cleared.

SHA256:

```text
baseline.so  87da4b011bcb9a59f9e776cc0034ba1ad51b0869f0922daff3d7f569a7bd0d84
candidate.so 7af8c2975395a68c20bacd5cc3fe22e955e105864c4ccb2479f37a0ec7403300
production.patch bf341c1c9fbbd6f39b3c32e50b0fb9aa297ca8f893bed1bae45f1e6ff40948c5
LDA reference af7a984e08d7a8bf12c3c2ef1e80fb574bed392b52ee31bb60d3f632c2456b00
PBE reference c9d665b0676d888ef450803f9d8e231873ae4922e744b6399c0f942edbedefcf
```

Raw `evidence/qualification-10402.json`, `qualification-10406.json`,
`candidate-profile-10405.{json,nsys-rep,sqlite}` and profiler CSV remain there.
Ignored local copies, scripts and logs live in
`.artifacts/pruning-gpu-20261019/optimization-*`; complete vectors and every
timing sample are retained rather than expanding tracked evidence quota.

## Revisit when

Requalify complete endpoints and independent numerical/resource gates before
changing the crossover, extending profiles/architectures, changing panel storage,
or claiming VV10 performance. Remaining costs include XC/AO work, partition
construction and more native production points than GPU4PySCF; none is assigned
a clean wall fraction by this change.
