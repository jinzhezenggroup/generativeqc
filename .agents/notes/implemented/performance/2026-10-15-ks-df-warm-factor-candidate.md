# Decision: opt-in, allocation-free hybrid KS warm-factor representation

Status: implemented opt-in candidate; issue-source endpoint qualification retained
Date: 2026-10-15
Related: #2474, #1567, #399

The opt-in policy below is superseded by the
[guarded default-replay decision](2026-10-15-ks-df-warm-factor-default.md).
The original qualification and its retained evidence remain unchanged.

## Problem

The local base `ab5282f74` has the same unconditional occupied-factor/frame reset
at `CudaKsPlan::Impl::begin()` described in #2474. `finish_legacy()` caches the
**current consumed physical density**, not its newly computed proposal. That is
the correct E/F/residual generation, but its canonical factor was discarded across
solves. The first separate warm K therefore selects dense exchange even when the
last-good density has a known exact occupied representation.

The issue's historical 96-atom M06-2X DF three-build observation does not establish
that this reset caused two additional convergence iterations. No such inference,
Direct-minus-DF timing subtraction, or attribution of #2464 XC work is made here.

## Decision

Keep `GENERATIVEQC_CUDA_KS_WARM_FACTOR=1` opt-in until the scientific and complete
endpoint gates are independently qualified. Store the exact density-generating
AO factor in the **existing warm slot** when the already-qualified final fitted
projection lease is available. Otherwise retain its ordinary density representation.

The candidate retains original AO coefficients in the existing intermediate frame
slot for fitted RKS. Its orthonormal-frame-to-AO recomputation at final handoff is
replaced by a direct copy, so the borrowed factor is exactly the one that produced
the consumed density, not merely a close recomputation or final-Fock frame. A frame
representation tag keeps the incumbent retention and handoff unchanged with the
candidate disabled; environment changes cannot reinterpret a previously stored frame.

Warm replay materializes D with the same density kernel and fixed occupations.
The source is an immutable prepared geometry/basis/auxiliary/metric/model owner;
cache owner/source/epoch/generation/occupation checks are required before borrowing
C for K. The cached factor has no dependence on mutable provider projection U.
The next build goes through the shared DF/RI-K facade and creates a fresh U lease.
Existing final-state token and force-handoff checks remain independent.

Frozen updates retain the original warm representation and generation, even if
intervening solves take several builds or accept a different current density.
Explicit/mixed/changed-geometry seeds do not borrow it. UKS, mixed precision,
truncated auxiliary rank and unavailable projection capacity retain dense first-K
fallbacks. No extra retained matrix or `O(naux * nao * nocc)` allocation is added.

## Alternatives not taken

- Retaining the most recent orbital frame without linking it to the warm density
  is unsafe when warm updates are frozen or an explicit seed changes the trajectory.
- Borrowing final canonical coefficients substitutes a nearby stationary density
  for the exact physical density consumed by the last K.
- Retaining another complete fitted projection would enlarge an already substantial
  provider lifetime and require a new resource reservation.
- Refactorizing arbitrary PSD seeds would repeat #399 work and introduce new
  reconstruction, rank, precision and synchronization gates. This candidate only
  reuses a factor whose producing density kernel is already known.
- Enabling an unmeasured optimization by default would bypass #2474's endpoint and
  independent energy/force acceptance requirements.

## Evidence and limits

Host-only executable tests run production cache/publication/restoration/export
methods with CUDA/provider doubles. They cover owner/source/device/stream/shape,
spin/rank/epoch/generation/precision mismatches, disabled replay, missing final
lease, failed publication, five frozen replays, and nonmutating density exports.
NumPy independently checks density reconstruction and dense-versus-occupied RI-K
algebra. Existing projection-lifetime tests exercise real facade guards for
truncated metric, unavailable projection capacity and interleaved scratch writers.
These tests are not real-device numerical or timing evidence.

The earlier host/source-contract selection reports **222 passed, 6 skipped**;
the final focused selection reports **206 passed, 6 skipped**. The benchmark's
separate host-double rerun reports 9 passed. Ruff, formatting and
`git diff --check` pass. A broader run also encounters 25 failures in the existing
incremental-resource tests because the current-name full native shared library is
not built; their failures are at library discovery, not a numerical/code assertion.
Local logs and before/after ccache statistics are retained under
`build/issue2474/`. That preliminary local compilation did not link a full
shared library; the separate remote qualification does.

Native CUDA tests exercise first-warm-K provenance, five frozen
replays, strict CPU-DF physical-state checks, stale final tokens, intervening
same-shape provider writers, explicit non-idempotent seeds, cold restarts, UKS and
changed geometry. Complete linked native runs on n1's RTX 5090 pass the restricted
and unrestricted cases. The independent CPU-DF matrix oracle differs by at most
`8.882e-16`; the two GPU K/Fock routes agree exactly on the focused fixture.
Seven actual first-warm-build audits also pass. These runs use finite Slurm jobs,
CUDA 12.9.1 and verified sccache 0.16.0 builds, not only header compilation.

The initial local-only resource diagnosis was incorrect: these machines have
independent Slurm controllers. SSH probes find allowed resources on n1, n2, n4
and n5. Finite Slurm probes confirm RTX 5090 driver access on n1 and n4; n2's
installed userspace driver disagrees with its running kernel module, so it is
used only for compilation. Device qualification runs on n1, without
bypassing Slurm or changing its assigned visibility. Independent three-atom DF
E/force references for M06-2X, PBE0 and B3LYP complete all 12 endpoints each.
The existing checkout still contains substantial unrelated dirty work and is
older than the issue-campaign source: its M06-2X implementation explicitly rejects
DF preparation. It must not be used to claim reproduction of the issue's M06-2X
branch. A separate shallow snapshot at exact issue source
`36574095876e4d12a3945b39cef72f6c77895f00` carries only the candidate C++/test patch
and the qualification runner. The original checkout and all unrelated changes
remain untouched. Baseline/candidate use one identical binary and switch only the
warm representation.

The current-source port preserves its compensated electronic-energy anchor.
Warm publication owns only seed representation and provenance; the convergence
controller still writes its incumbent energy/compensation pair. Host probes
explicitly verify that publication does not reinterpret or overwrite that anchor.

On the older checkout, PBE0/def2-SVP plus the explicit cc-pVDZ-JKFIT auxiliary
passes all 12 baseline, 12 candidate and 12 separately audited E/force endpoints.
Maximum energy/force errors against independently reconverged matching DF
references are `1.024e-12 Eh` / `1.348e-11 Eh/Bohr`. Ten actual first-warm K/Fock
comparisons have conservative error bounds `1.96e-14` / `2.5e-15`. Initial missing
CUDA library and packaged-force-artifact failures remain retained, as does a B3LYP
baseline final-state export refusal; these are not silently filtered into an
accepted population. No B3LYP claim is made from that older checkout.

The complete source-matched campaign is now retained in
[`benchmarks/results/ks-df-warm-factor-20261015/`](../../../../benchmarks/results/ks-df-warm-factor-20261015/README.md).
One identical library (`b2d38025066fcece1afdb23f0dd1e14879208bc51fe7b2d490ff32e68507bc9c`)
uses switch 0/1 for baseline/candidate. The measured full tracked-source patch is
`441bab799cb16bae9646fdc0f51eee6c1ddee7c6b74efbc6ebc281a8502262ab`.
The issue's 96-atom M06-2X DF three-build frozen-warm branch is reproduced, not
inferred from historical Direct/DF timings.

All **216 clean native E/force endpoints** pass independent matching DF references
at 3/48/96 atoms, three functionals and both variants. The nine reference
populations contain 108 complete calls. Maximum clean energy/force errors are
`3.638e-11 Eh` / `1.635e-10 Eh/Bohr`. Another 180 diagnostic and 66 interleaved
native endpoints pass their same reference gates. Thirty actual first-warm matrix
audits have conservative raw K/Fock error bounds `8.243e-13` / `2.154e-13`, below
the unchanged `2e-11` matrix gate. Extra audit K calls, host storage and fences
are excluded from clean times and the ordinary semantic-work ledger.

96-atom complete frozen-warm medians, five calls each:

| Functional | Baseline / candidate seconds | Fock builds |
| --- | ---: | --- |
| M06-2X | 36.562 / 21.403 | 3 / 1 |
| PBE0 | 22.821 / 12.501 | 5 / 1 |
| B3LYP | 20.498 / 13.206 | 3 / 1 |

All three moved-warm populations also reach one build, while cold/moved build
counts remain unchanged. Separate same-owner ABBA measurements hold the exact
factor-backed seed and compensated energy anchor frozen while switching first-K
policy only. Five samples per arm and geometry pass the repository's robust
noise assessment; original-geometry reductions are 42.31%, 45.44% and 35.62%.
These are complete E/force times, not iteration-normalized estimates.

The generation-linked factor is the original producer of consumed D, not the
final-Fock canonical proposal. On the measured M06-2X branch, dense first K changes
the compensated electronic energy by `3.325224230570919e-12 Eh`, failing the
existing `1e-12` energy gate. Its second build also fails that gate. Restoring
the original factor reproduces the incumbent occupied contraction, yields zero
first energy change and passes the ordinary fresh F[D]/density/residual audit.
No controller gate or compensation bookkeeping is changed.

A representative traced warm pair separates dense-to-occupied first K
`4.009605 -> 0.548070 s` on the GPU-event clock from two additional baseline
physical builds costing `6.001204 + 6.010354 s`. The retained additive host ledger
also accounts for both complete E/force calls, explicitly leaving uninstrumented
work as a remainder. Provider roots are nested in KS J/K, never added again;
host and GPU clocks are alternative views. This explains the measured convergence
branch without crediting unrelated XC work or subtracting a Direct endpoint.

The 48-atom ordered/tiled provider lacks the complete retained projection required
by this conservative factor certificate. It stays dense on first warm K and
one-build warm, with no claimed speedup. Both 96-atom M06-2X variants retain the
same 175,688,520 state, 92,808,608 XC and 14,299,355,893 provider bytes. No numeric
arena or unconditional full U allocation is added. Explicit owned KS synchronizations
and diagnostic profiler fences remain separate; no global allocator peak or
exhaustive CUDA-library synchronization count is inferred.

The current-source focused PBE0 RKS/UKS device regression also passes against the
same library: reported K/Fock/CPU-DF errors are `2.22045e-16` / zero / `6.66134e-16`.
Its inherited bare-PBE-plus-quarter-K fixture was invalid under the current model
validator, so the test now uses the matched 0.75 semilocal exchange scale and
independently rebuilds scaled CPU PBE0. Production validation was not relaxed.
The corrected standalone fixture was linked without changing the frozen library.
An attempted transfer to n4 failed at host glibc compatibility, before valid
native endpoints; it is retained separately, not counted as accepted evidence.

## Revisit when

Keep the candidate opt-in when integrating into a different source/build or
considering default promotion. This qualification does not repair the older
checkout's unrelated M06-2X availability or B3LYP export refusal. Additional spin,
precision, metric/rank and lifetime domains need their own gates; do not remove
bounded fallbacks or skip physical audits to obtain one-build replay.

Smaller/tiled layouts could eventually expose a separate shared factor certificate
that does not require complete U residency. That is a new ownership/qualification
decision, not permission to allocate another full projection or bypass the current
lease, full-metric and capacity checks. Preserve the reviewed source reconstruction,
failure history, numerical gates and complete-endpoint comparison when revisiting it.
