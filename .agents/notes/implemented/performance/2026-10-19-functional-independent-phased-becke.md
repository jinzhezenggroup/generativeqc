# Decision: functional-independent native phased Becke admission

Status: implemented; qualification scope and limitations below
Date: 2026-10-19

## Problem

The initial [LDA/PBE optimization](2026-10-19-native-pruned-phased-becke.md)
addressed the [pruned endpoint diagnosis](2026-10-19-pruned-endpoint-attribution.md)
but restricted its generated scheduling policy to two functional names. Becke
partition response depends on geometry and partition cotangents, not on the XC
functional that produces those cotangents or on the number of spin channels.
The whitelist was therefore the wrong ownership boundary.

This note supersedes that initial note's profile restriction and qualification
snapshot, not its original measurements. The earlier source/binary identities
and diagnostic evidence remain historical evidence rather than being relabelled.

## Decision and invariants

- Emit the same existing 48-atom crossover for every native compiled profile.
  There is no LDA/PBE, hybrid, meta-GGA or spin whitelist in the schedule.
- Reuse the existing phased equations and executor, including the 128-atom cap,
  full-point-lane, retained-center and actual-device sm_120 prerequisites.
  Unsupported hardware and optional allocation misses keep the bounded path.
- Charge phase panels only to the authoritative cached capacity. The mandatory
  minimum is unchanged. Exact cached, one-byte-short, minimum and reduced-lane
  cases must preserve byte admission and typed failure cleanup. At 96 atoms and
  a 256-point tile the optional reservation is 39,755,392 bytes per force owner.
- Do not expand public scientific admission. In particular, compiled M06-2X
  bindings are not sufficient to admit large split-hybrid native force contexts.
  Those contexts keep their packaged Python consumer and its already-generic
  phased geometry. Composite/nonlocal owners likewise retain their existing
  generic policy; no production CPU/reference work is added.
- Do not change XC equations, precision, derivative formulas, screening, SCF
  stopping criteria, grids or pruning defaults. Mixed geometry remains on its
  existing executor. Generic scheduling is not blanket pruning qualification.

## Source-matched complete endpoints

The final pair is built from clean master `2434d79b4`, with only the runtime and
emitter changes applied to the candidate. Both contain all twelve native
profiles. RTX 5090, CUDA 12.9, sm_120, Release, Direct STO-3G, one host/BLAS
thread. Each sample constructs a fresh Calculator, prepares, converges SCF,
computes analytic forces, destroys the owner, collects garbage and synchronizes.
Process/artifact caches are warm, but no density is reused between samples.
One complete warmup per binary precedes six alternating samples per method.

Slurm job 10414, unchanged assigned visibility `0,1,3,4`, local device 3
(physical GPU 4, idle at entry), default qualified pruned production grids:

| 96-atom method | Baseline median (range), s | Candidate median (range), s | Speedup |
| --- | ---: | ---: | ---: |
| LDA-RKS | 21.092875 (21.086356–21.102225) | 14.672544 (14.642531–14.684720) | 1.4376x |
| PBE-RKS | 15.240628 (15.233226–15.253408) | 11.130099 (11.126033–11.179090) | 1.3693x |

Both engines here are GenerativeQC, not a new GPU4PySCF ranking. Energy deltas
are zero; maximum force deltas are 1.24345e-13 and 1.03917e-13 Eh/Bohr.
All samples retain 15 Fock builds and respectively 2,467,200 / 1,579,008 points.
Independent GPU4PySCF dense level-9 references and level-8/9 stability margins
from jobs 10323/10324 pass the original production gates:

| Method | Energy including margin / gate, Eh | Force including margin / gate, Eh/Bohr |
| --- | ---: | ---: |
| LDA | 2.1271441e-5 / 3.2e-5 | 1.4289195e-5 / 5e-5 |
| PBE | 4.3116846e-5 / 4.8e-5 | 1.2171169e-5 / 3e-5 |

## Cross-family diagnostic endpoints

Job 10416 uses the same complete-endpoint protocol at 48 atoms with an explicit
**fixed** `GridSpec(24, 8, 16)`, 147,456 points. This is a controlled schedule
comparison, not independently qualified production-grid accuracy or a README
ranking. Hybrids require an explicit grid under their existing public policy;
the optimization does not invent a new default grid for them.

| Method | Baseline median (range), s | Candidate median (range), s | Speedup | Fock builds |
| --- | ---: | ---: | ---: | ---: |
| r2SCAN-RKS | 3.069648 (2.955094–3.308718) | 3.120398 (2.995783–3.244687) | 0.9837x | 14 |
| PBE0-RKS | 1.299137 (1.295594–1.302580) | 1.243876 (1.242576–1.292645) | 1.0444x | 14 |
| B3LYP-RKS | 1.339283 (1.338106–1.346845) | 1.286657 (1.283722–1.309953) | 1.0409x | 14 |
| PBE-UKS | 6.024772 (5.796684–6.218729) | 6.101943 (5.640389–6.302459) | 0.9874x | 31 |

All E/F parity and unchanged-grid/Fock-work gates pass. Maximum energy/force
deltas across these cases are 1.36425e-12 Eh / 1.86518e-13 Eh/Bohr.
UKS water is an equal-spin timing control; the derivative fixtures below use
genuinely polarized states. The two slightly slower/noisy medians stay visible:
functional independence does not establish a universal endpoint speedup.
No functional-specific scheduling exception is inferred from this small-grid
population. There is no M06-2X or VV10 speedup claim in this change.

## Qualification and unsuccessful attempts

The generated host contracts execute real capacity/admission arithmetic for all
twelve profiles using their actual source and spin metadata. They exercise
exact/short/minimum budgets, reduced lanes, invalid shapes, typed errors and
cleanup with inert CUDA allocation boundaries rather than source fingerprints.
The final focused host suite passes 163 cases, with 37 allocation-gated cases
skipped outside Slurm. All applicable staged pre-commit checks pass, including
compiler/native/CUDA ownership, public contracts, resource-policy inventories,
structural lint and Python/CUDA formatting. No generated ownership report is
committed because semantic ownership is unchanged.

The allocation-gated matrix uses 48/96/128 atoms, original and moved geometries,
reconverged directional energy differences at 2e-4 and 1e-4 (1e-7 derivative
gate), and fresh-owner parity (1e-9 energy / 1e-7 force gates). Admitted native
profiles also force the published minimum tier. UKS fixtures replace one H by
He to make a genuine doublet HeH fragment without degenerating every separated
H2 fragment into an open-shell triplet. M06-2X checks the existing fail-closed
native admission and live phased counters on its packaged fallback instead.

Job 10418 passes all 37 GPU cases: the twelve profiles at three sizes and the
existing WB97M-V composite regression, including both semilocal and nonlocal
owner phase counters. Assigned visibility is `0,1,3,4`; local device 3 selects
the idle physical GPU 4 without altering that visibility. Job 10419 separately
passes the 48-atom r2SCAN-UKS native and M06-2X-UKS packaged-fallback cases under
both memcheck and synccheck (two cases per tool), each reporting zero errors.
That job uses assigned visibility `5`, local device 0. All real-GPU execution
uses finite `srun` jobs on n1, never the maintenance node.

Initial job 10408 incorrectly expected compiled M06-2X graphs to be publicly
native-admitted. Jobs 10409/10410 exposed existing explicit-hybrid-grid and
missing-packaged-fallback prerequisites. Job 10415 reached M06-2X forces but
the new test queried a nonexistent composite-only `last_work` field. These are
retained unsuccessful attempts, not numerical passes. The corrected test uses
the existing source metrics. The ordinary source-matched M06-2X s/p AOT
artifacts were built through verified sccache without changing admission.
Job 10417 selected a busy assigned GPU and failed preparation with
`std::bad_alloc`; it is excluded, not treated as numerical evidence.

## Provenance and reproduction

On n1, the retained capsule is
`/home/jzzeng/generativeqc-qualification/pruning-optimization-20261019/`;
this extension's outputs are under `evidence/general-20261019/`. Both CXX and
CUDA compiler commands invoke verified sccache 0.16.0, with before/after stats
retained and no cache clearing. Python fallback packaging is a separate build
target; it does not replace either source-matched native binary.

```text
baseline.so  82bd7780d33714c8fc58bcd9eb554f5f79d68834e3cb5542854b40b7af9f7d33
candidate.so a4526e20abc1bffaec9dcb6bb78bf84137ee5369dfefa9931dbfb65ec8c85938
production.patch 1442dc772d6dfd4683f7b0cab212a3edd2e6cfc68488d63423b8400176338dbf
LDA reference af7a984e08d7a8bf12c3c2ef1e80fb574bed392b52ee31bb60d3f632c2456b00
PBE reference c9d665b0676d888ef450803f9d8e231873ae4922e744b6399c0f942edbedefcf
```

The capsule retains source archives, patch, separate binaries, build/qualification
drivers, manifests, complete vectors, all timing samples and unsuccessful logs.
Ignored local copies live in `.artifacts/native-phase-general-20261019/`.
Run `tests/python/test_native_stationary_phased_budget.py` for host contracts;
inside a finite Slurm allocation, set `GENERATIVEQC_DFT_CUDA_TEST=1` and supply
the matching native library plus packaged stationary M06-2X artifacts for the
GPU matrix. `GENERATIVEQC_TEST_DEVICE_ID` selects within, never overrides,
Slurm's assigned visibility. The existing composite regression additionally
requires `GENERATIVEQC_TEST_WB97MV_CUDA=1`.

## Rejected alternatives and revisit conditions

Reject a functional-name whitelist, forcing native admission just because a
binding was linked, reducing a method's grid without an independent reference,
and choosing only favorable timing populations. Keep the bounded fallback and
actual-device checks rather than making optional pair panels mandatory.

Requalify complete endpoints, moved forces, resource tiers and independent
accuracy gates before changing the crossover, panel storage or hardware domain,
or claiming broader production-grid, M06-2X or VV10 speedups. Small-grid launch
overhead is a possible future shape-based tuning question, not evidence for a
new XC-specific exception.
