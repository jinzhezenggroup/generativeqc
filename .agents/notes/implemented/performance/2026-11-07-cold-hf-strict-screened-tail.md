# Decision: promote bounded cold HF with a strict physical target tail

Status: implemented
Date: 2026-11-07

## Problem

The user requested remaining HF/DFT cold-endpoint improvements, not another
WB97M-V-only kernel experiment. The historical 2026-10-26 screened HF default
was contained after its default-request force error reached 3.02e-7 Eh/Bohr,
against the unchanged additional 1e-7 diagnostic. Tighter-request qualification
did not qualify that failed default. Preserve those original failures in
`benchmarks/results/cold-hf-defaults-20261026/` and the negative-evidence inventory.

The subsequent standard-owner stopping policy permits a 1e-5 AO maximum for
ordinary default requests. Using that ceiling after a changed proposal trajectory
does not establish the stronger independent force accuracy. The proposal must not
publish its temporary operator or inherit its DIIS/energy history as a target.

## Candidate

Automatically admit only the existing large, closed-shell, single-system,
through-d, cold bounded-Direct domain. Retain its at-most-eight FP64 proposal
steps at 8e-8 screening, explicit target DIIS/energy reset, and single total user
iteration allowance. During the physical target tail, withhold standard-owner
residual relaxation: require the existing strict `min(density_tolerance, 1e-8)`
AO maximum. This tightens acceptance; public tolerances, target screening and
precision are not relaxed. Ordinary non-proposal and provisional gates remain
unchanged. Warm/imported owners never inherit the temporary graph.

Unset admits eligible proposals; `0` and other non-`1` explicit values retain the
ordinary path. Scratch admission is a separate HF-owner decision: the first
strict-tail candidate kept incumbent pages; the refined candidate admits larger
bounded reusable task pages in the qualified domain described below. There is
no AO/jet cache, GPU-name dispatch, production PySCF work, or new CUDA arithmetic.

## Evidence and decision boundary

The single source0644bc07 / ELF2dd1887b A100 screen, LSF227290, completes default
HF and PBE0 cold plus three own-density warm calls. It reports HF17.248835s cold /
2.370330s warm median versus fresh GPU4PySCF21.621229s /1.506339s; PBE0 is
13.357153s /3.828216s versus14.039905s /3.168151s. These are one-owner screens,
not a cold statistical ranking or proof that every method remains behind.
The first reference cold population may include first-use artifact costs; it is
retained rather than replaced with a faster later observation.

Native r2SCAN-3c completes17.346308s /6.943703s, but its reference fails loading
`libmvec.so.1` before producing a complete endpoint. It has no reference latency.
Repair the private reference runtime without changing science before claiming
a current r2SCAN-3c gap. The failed job remains exit1, not an accepted cohort.

The first strict-tail candidate source
`993e79b24f60511370da9a3d4bb77c9182d4cb870f3d938c4eff035b3570c641`
includes master0a178776c and the reviewed composite-source dependency. A normal
SM80+SM120 Release build uses verified sccache0.16.0 and restores the borrowed
source/runtime on exit. Candidate results must bind the actual new ELF and must
not reuse the screen's old source identity.

Before retaining automatic admission require unchanged strong independent HF
energy/force gates, actual default-request cold/moved/warm/checkpoint behavior,
bounded page-refusal and short-allowance controls, clean same-image rollback/default
complete endpoints on A100 and physical5090, unchanged DFT negative controls,
and complete work/provenance. Reject the candidate if stricter convergence
eliminates its cold benefit or protected-card behavior regresses. Do not leave an
unqualified new automatic default or describe a host/build pass as GPU evidence.

## References

- External engineering memory: issue2976, plan comment6703840874.
- `tests/python/test_hf_cold_screening_cuda.py`.
- `tests/native/test_precision_policy.cpp`.
- `.artifacts/hf-dft-cold-20261107/` and
  `.artifacts/hf-cold-strict-target-20261107/` in the managed follow-up worktree.

## Refinement after the first dual-card screen

The source993e79b2 candidate passes five independent/default/lifetime/allowance
tests on each card. Its default-request force error is 1.55e-9 Eh/Bohr, safely
below the unchanged 1e-7 diagnostic. However the strict tail increases total
Focks from eleven to fifteen. With incumbent pages the complete A100 cold gain
is only 17.228283 to 16.686511 s (LSF227478), while5090 is 18.168072 to
18.351221 s (Slurm16661). These are single-owner screens, not a statistical win.

An isolated explicit-page screen in the exact same strict-tail image yields
A10016.109314 s cold /2.263241 s warm median (LSF227514), and509016.704658 s /
2.670403 s (Slurm16669), with the same fifteen/one Focks and strong independent
observables. The first proposal's blanket refusal to expand scratch therefore
leaves measurable pagination overhead. This is not the rejected multi-gigabyte
AO/jet cache: it is an existing reusable, allocation-ledger-bounded task page
with a pre-upload retry at the incumbent ceiling.

The revised candidate admits the existing 32M-task/6-GiB page only for large
singleton through-d FP64 RHF bounded owners with a sufficient iteration allowance.
Small, batch, UHF, higher-l, mixed, incremental and physical-reference owners
retain the incumbent default. Generic Direct/KS resource policy is unchanged.
Targets below24GiB retain the1/16 budget; larger targets may use1/4, capped6GiB.
Seed history does not select an approximate operator or prevent exact page reuse.
The helper is an HF-owner policy, separate from generic kernel/header contracts.
Do not relabel these explicit1 measurements as default results.

## Combined trial and final-image boundary

Actual LSF 227637 and Slurm 16671 qualify the combined HF policy plus a DFT
relative-response reassociation. HF cold improves 17.555007 to 16.466193 s on
A100 and 18.128387 to 16.878941 s on 5090. Each card passes six independent
HF/default/lifetime/allowance/page-refusal tests; default force error remains
below the unchanged 1e-7 gate. Cold Focks remain fifteen versus the ordinary
eleven; warm Focks remain one. The DFT math is independently correct but yields
no material endpoint benefit, so it is rejected and removed. See the separate
[rejection](../../rejected/2026-11-07-density-force-relative-map.md).

The final HF-only candidate includes master `f27bdb3b0`, including the auxiliary-g
correction. Its source is
`318c580fcb985361a13af117880225209ddb4b25202efab807b9f6439ac4a707`, ELF
`d2f3353d85677f1e9bb505c7e0dc4fa3ae6f6fe7a7a74d84fb999c230f3e2be1`.
The normal Release SM80+SM120 build passes and restores the borrowed runtime;
its verified sccache0.16.0 receipt reports 95.02% cache hits. The combined trial
and earlier explicit-page experiments do not substitute for this image's gates.

## Final automatic-default qualification

Actual LSF 227737 and Slurm 16672 both finish exit0 with the distinct final
HF-only source/ELF. Each card passes all six independent HF/default/moved/warm/
checkpoint/allowance/page-refusal cases and sixteen signed Libcint J/K cases;
the raw-AD/numerical-fallback probe passes. 5090 memcheck and racecheck report
zero errors/hazards. No public numerical control or independent gate is loosened.

| Card | Ordinary cold | Automatic cold | Ordinary warm median | Automatic warm median |
| --- | ---: | ---: | ---: | ---: |
| A100 40 GB | 17.591863 s | 16.449529 s (-6.49%) | 2.402655 s | 2.303642 s |
| RTX 5090 | 18.130591 s | 16.878195 s (-6.91%) | 2.906185 s | 2.690658 s |

The complete endpoint includes energy, analytic forces and host export. Each
arm has one retained first cold and three frozen own-density warm calls. Cold
Focks increase eleven to fifteen because the inexpensive screened proposal
changes the trajectory and the strict tail does more physical work; warm Focks
remain one. These are reproducible one-owner acceptance screens, not statistical
rankings or a universal HF speedup claim. PBE0, r2SCAN-3c and WB97M-V retain
their physical/operator work and all cold/warm ratios remain below1.03. No DFT
speedup or complete GPU4PySCF superiority is claimed.

The default candidate's maximum independent force error across the four exports
is 1.548e-9 Eh/Bohr on each card, versus 5.101e-8 for the ordinary route, under
the unchanged 1e-7 gate. Maximum absolute energy error is 1.409e-9 Eh, below1e-8.
HF's pre-existing fine-grained precision inventory remains incomplete in both
arms, while its full Fock-stage census is valid. Preserve those false inventory
flags; this qualification does not deliver a new primitive precision audit.
DFT negative controls retain complete inventories.

Promote unset selection in the admitted HF domain; retain explicit rollback,
short-allowance refusal, conservative unsupported-domain policies and bounded
allocation retry. The previous failed default-force and slower5090 small-page
histories remain immutable. They are superseded, not erased.

Source-bound records, older strict-tail/explicit-page populations, rejected DFT
patch and independent repaired r2SCAN reference are retained in
`benchmarks/results/hf-cold-defaults-20261107/`. Its CPU-only replay rejects
source/ELF relabeling, missing populations, changed controls/work, numerical
failures, protected regressions and promotion of the rejected DFT trial, including
under `python -O`. The A100 original final archive SHA256 is
`8ba71b842fac9ffeea988e30ba405af23e51303d631bf09fe390454952ce111b`.

This supersedes the containment decision in
[the original note](2026-10-26-cold-hf-screening-pages.md), not its retained
negative evidence. Revisit scope or resource limits only with complete endpoints
and independent observables on both protected GPU classes.
