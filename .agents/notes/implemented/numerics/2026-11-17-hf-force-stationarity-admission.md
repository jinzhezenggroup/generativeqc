# Decision: separate HF force stationarity from energy-only stopping

Status: implemented
Date: 2026-11-17

## Problem

Issue #3092 retains an independent Cartesian water/def2-TZVP force failure at
the initial geometry with stock Direct CUDA RHF controls. Both auto and portable
AOT profiles on RTX 5090 and RTX PRO 6000 Blackwell report convergence while
exceeding the unchanged `1e-7` maximum-force gate. Those observations do not
isolate a mathematical kernel defect or an SM120-specific cause.

The default orbital-gradient policy admitted force endpoints even though
`standard_hf_stationarity_owner` already excluded them from relaxed AO gates.
The orbital-policy flag bypassed that exclusion: both the iteration predicate
and the retained-Fock force publication accepted the energy-oriented
`sqrt(energy_tolerance)` gradient threshold. A warm endpoint could then reuse
the insufficiently stationary determinant without further refinement.

A finite n1 diagnostic on the previously retained source
`038aedeccf3964b58acec499cc9684f423a7a93f` changes only the convergence policy,
not any public tolerance, basis, arithmetic or force gate. For both profiles,
the initial error changes from about `6.758e-7` (9 iterations) to `1.386e-9`
(12 iterations). Moved force errors also pass. This establishes the stopping
policy as a sufficient cause for that reproduced failure, not a general
force-error theorem for the strict residual threshold.

## Decision

The shared Direct/DF convergence resolver receives the requested observable
ownership explicitly. Force endpoints select the existing strict physical
AO-maximum and density-change conjunction. Explicit `orbital-gradient` requests
on force owners fail closed, like other unsupported orbital-policy owners.
Energy-only eligible owners keep the existing orbital-gradient policy. Public
energy, density and screening defaults remain `1e-9`, `1e-5`, and `1e-12`.
No card, SM, basis name or geometry participates in this decision.

Compact DF iteration and final-state selection must also withhold their relaxed
AO ceiling for force owners. Changing only the orbital selector would otherwise
leave DF force publication at a different physical stationarity level. Physical
reference, UHF, ECP, mixed-arithmetic and application-budget protections remain.
Prepared execution already binds `compute_forces` to its captured identity;
energy-to-force transitions must re-admit the native owner, not reuse a looser
energy proof as a force proof.

## Tradeoff and validation

Force endpoints can require more iterations; no speedup or unchanged benchmark
ranking is claimed. This narrows the historical Direct/DF promotions below;
their source-distinct measurements and former acceptance budgets stay intact.
The strict residual remains a solver criterion, not a universal guarantee of
independent force accuracy on ill-conditioned or differently screened systems.

Executable host probes cover default/strict/explicit requests and force-owner
admission. The independent device regression covers Cartesian RHF and UHF,
SVP/TZVP, cold/frozen-warm/moved/energy-to-force calls, and matching fitted RHF
operators. The oracle runs in a separate process; production imports are
forbidden. The original `1e-9` energy, `1e-7` force and `1e-7` translation
acceptance gates remain unchanged. Actual device receipts must identify the
source, binary, profile, physical device and finite Slurm allocation.

## Retained device evidence

The [source-bound qualification](../../../../benchmarks/results/hf-force-stationarity-20261117/README.md)
retains successful finite Slurm jobs 17670 (RTX 5090) and 5324 (RTX PRO 6000
Blackwell), four fresh processes per card/profile, identical ELF bytes across
cards for each profile, and independent raw reference arrays. All 96 cases /
576 endpoints, including 480 E+force endpoints, pass the original gates.
Maximum energy/force/translation errors are `1.1221e-10`, `4.8488e-9` and
`2.2471e-13`, respectively. The original TZVP cold/warm case passes all 32
observations at maximum force error `1.3860e-9`; an earlier-source negative
regression still fails at `6.7581e-7`. Both cards are SM120; no cross-SM or
performance qualification follows from these results. Replay recomputes the
errors and DF finite differences rather than trusting stored pass flags.

## Integrated evidence and performance acceptance

Master reconciliation `e9f3031ae7f324111ad5c84cf1549ac4a75f4bb6` changes the
native source identity to
`1615b290a1e7d706a1e360036b410d490d73b1835d5ac77f1de1ef4d937d8d40`.
The retained bundle therefore adds independently acquired observations rather
than assigning the preceding measurements a new label. Fresh finite jobs
17672 (n1) and 5325 (n2) pass another 576 endpoints / 480 force endpoints;
maximum errors are `1.1220890883123502e-10` energy,
`4.8487097220302824e-9` force and `2.3581137043038325e-13` translation.
The complete 246-test host error/retry suite passes in job 17683.

The user explicitly requires **no performance regression**. Scientific success
does not discharge that acceptance condition. Strict force stopping can add
iterations, so PR #3108 remains Draft pending complete-endpoint paired timing
against source-matched master `8f4166f0c6a692045eaf568c814b868aa1057482`.
Retain originally correct old endpoints in that comparison; exclude incorrect
old force results from valid speed rankings without hiding their failures.
Do not weaken stationarity, oracle gates or public tolerances to disguise work
amplification. More iterations are a diagnostic, not a substitute for timing.

## Negative performance result: strict-only repair is not accepted

Finite complete-endpoint jobs 17686 (RTX 5090) and 5329 (RTX PRO 6000 Blackwell)
compare source-matched master `8f4166f0c6a692045eaf568c814b868aa1057482` to the
integrated repair, auto and portable, ABBA/BAAB fresh-process ordering. All
1,152 observations are retained and independently replayed. Every one of the
576 repair observations passes science, but performance acceptance fails on
both cards, including endpoints whose old results were already correct.

Direct RHF moved/SVP is 23.17%/26.03% slower on 5090 auto/portable and
25.40%/24.90% slower on PRO 6000 auto/portable. PRO 6000 auto moved/TZVP is
21.53% slower; matched DF moved/SVP is 11.36% slower. Direct moved work grows
from 8 iterations / 10 Fock builds to 12 / 13; DF grows from 8 / 9 to 12 / 13.
The PRO 6000 auto/SVP moved 95% paired-bootstrap ratio interval is
`[1.25147, 1.25751]`. The original invalid TZVP cold/warm samples are separately
unrankable; their failure does not excuse these valid-endpoint regressions.

Consequently PR #3108 remains Draft and this implementation is a scientifically
validated **prototype**, not an accepted complete fix. Do not relax force gates,
raise a stationarity ceiling by fitting these six fixtures, drop slower valid
cases, or restore energy-only force publication to meet a speed target. A
follow-up must reduce strict-stationarity work or its endpoint cost while
preserving complete force qualification and explicit bounded fallbacks. Neither
a microbenchmark nor an opt-in tuning winner satisfies that stock-path gate.

The campaign also caught a build-provenance trap: replacing newer sources with
an older git archive at the same build path can leave Ninja objects and the
generated identity newer than all inputs. The first supposed master binary
still had the candidate identity and bytes; allocation 5328 was stopped and
its partial records retained as rejected setup. Refreshing the four genuinely
content-different native inputs (without changing bytes or compiler-cache keys),
then rebuilding and asserting independently computed source identity in job
17685, produces the real baseline. Future reused-build comparisons must verify
actual ELF/source identity, not infer it from a successful configure/build exit.

## Comparator follow-up: do not infer a stricter default contract

After the user asked whether GPU4PySCF supplies the same default accuracy,
finite RTX5090 jobs 17690/17691 execute 108 complete endpoints across stock
GPU4PySCF 1.8.1, an explicitly tighter comparator and both native prototypes.
Exact Cartesian direct-RHF water SVP/TZVP fixtures and the independent CPU
oracle are unchanged. Three fresh stock comparator chains per basis give
maximum cold/moved force errors `1.12874e-6`/`5.93425e-7` (SVP) and
`1.29161e-7`/`5.67409e-7` (TZVP), despite all energies passing `1e-9`.
Only three of 30 stock comparator force endpoints pass `1e-7`; every result
and failure is retained. Explicit `conv_tol_grad=1e-9` instead passes every
comparator endpoint, maximum force error `2.28321e-10`.

Thus master and GPU4PySCF stock settings have comparable error scales on
these examples, not identical force accuracy. GPU4PySCF is about five times
more accurate for the original TZVP cold geometry; native master is more
accurate for the moved geometry and SVP. The observation does not establish
a card/SM120 mathematical defect or a universal default force-error guarantee.
It also does not justify forcing the 20--26% slower strict prototype onto the
default path solely to exceed comparator accuracy. Keep the unmerged prototype
Draft and retain its failed master-relative performance gate. The user's new
comparator-based requirement must inform an explicit accuracy-contract decision,
not silently relabel failed `1e-7` samples as passes or erase earlier evidence.

Complete endpoint times/work are retained but the installed comparator uses
CuPy contractions without cuTENSOR. The tighter setting is a diagnostic, not
a minimally accurate performance optimum. Do not promote a speed claim from
that environment or generalize these two direct-RHF cases to DF/UHF/other GPUs.
See the comparator section and independently replayable `gpu4pyscf.json.xz`
in `benchmarks/results/hf-force-stationarity-20261117/`.

## Matching the DF oracle

Do not compare different fitted operators under the direct-HF energy gate.
The shared PySCF basis exporter defaults to spherical auxiliary functions,
whereas a Cartesian PySCF molecule inherits Cartesian auxiliary functions.
Even after matching that representation, stock PySCF Cholesky DF is not the
native default spectral metric with relative cutoff `1e-10`. That cutoff is
gauge dependent: native Cartesian components have unit self-overlap, while
raw PySCF Cartesian components need not. The initially mismatched comparisons
are retained as failed setup observations, not production acceptance.

The DF reference therefore constructs its metric and three-center integrals
independently with PySCF/CINT, normalizes auxiliary components by independently
computed self-overlaps, and retains the published default spectral cutoff.
No native matrices, density seeds or fitted factors enter the oracle. Complete
forces come from reconverged displaced energies, including retained-projector
and auxiliary-center response. Two five-point steps (`0.01`, `0.02` Bohr) must
agree within `5e-8`; their Richardson extrapolation supplies the independent
reference. Displaced energies and unchanged metric ranks are retained. Public
production controls and the `1e-9`/`1e-7` acceptance gates do not change.

This explicitly fitted auxiliary model qualifies the shared force-stationarity
owner, not every auxiliary selection, fit error, or default DF endpoint.

## Superseded scope

- [Direct orbital-gradient default](2026-11-10-default-hf-orbital-gradient.md)
- [Compact DF orbital-gradient default](2026-11-11-df-hf-orbital-stopping.md)

Revisit force admission only with a separately qualified derivative-stationarity
criterion and independent complete-endpoint evidence, not a matching energy
threshold, a microbenchmark, or a single GPU label.
