# Proposal: budget-selected Fast across the README endpoint matrix

Status: proposed
Date: 2026-10-14

## Objective and authority

The user requested a staged PBE0 PR followed by Fast support for every README
benchmark method. PR #2452 preserves the first milestone. The method inventory
is `benchmarks.readme_dft.METHODS`, not a separate functional whitelist. Direct
and supported DF Hamiltonians, their auxiliary metric response, moving grids,
nonlocal terms and composite corrections must remain unchanged.

## Acceptance

Accepting the Fast preset and silently executing two strict endpoints is not
support. Require positive, execution-owned FP32 work and explicit returned
arithmetic/fallback provenance. Independently check energy, maximum Cartesian
force component and RMS force against the requested allowance. Qualification
is not a mathematical per-call certificate; explicit targets still request
their real-cost audit. Do not waive failed gates to enlarge method coverage.

Compare complete cold, frozen-seed warm, moved and moved-warm endpoints with
the same optimized ordinary FP64 controls. Include initial guesses, correction
work, retries and synchronization. Keep all failed/incomplete populations and
semantic work counts; no faster-kernel claim substitutes for endpoint timing.

## Implementation sequence

1. Requalify the first milestone against current master, particularly native
   Hückel seed preparation and shared PBE subnormal response behavior.
2. Extend typed XC lowering at the shared scalar-graph owner. FP32 parameters,
   constants, intermediates, elementary functions and results must all use
   single precision; widening a completed FP64 result is not an implementation.
   Preserve strict source and mathematical semantics by default.
3. Share the precision request across HF/KS preparation and final derivative
   consumers instead of encoding generic precision requirements in a PBE-only
   descriptor. Preserve stable public descriptors and qualified seed behavior.
4. Admit each complete direct/DF endpoint only after its numerical, lifetime,
   resource and timing gates. Retain FP64 for ill-conditioned/tail work and
   unsupported components, and report that boundary rather than claiming all-FP32.

## Current boundary

Direct PBE, PBE0, B3LYP and r2SCAN have now passed complete public singlepoint
qualification within the bounded domain in the application-precision guide.
Slurm 8287 includes the shared optimized FP64 comparator; its 1.03--1.22 ratios
are complete singlepoint measurements, not README prepared timing. Other typed
semilocal implementations remain compiler candidates, not automatic admission.
Point checks do not qualify a complete trajectory, DF metric response,
RSH/VV10, D4/gCP, Hessian, TS or dynamics.

## Follow-up integration constraints

The typed graph/guard implementation and its failed polarization experiment are
recorded in `../implemented/numerics/2026-10-14-typed-semilocal-fast-candidates.md`.
PBE0 milestone PR #2452 is separate; do not silently grow it into an unqualified
all-method public promotion.

The implemented v2 allowance transport extends the method descriptor privately,
independently of the legacy KS extension. HF correctly rejects KS options but
can receive the new immutable, versioned request without enlarging the stable
public descriptor. Preparation copies the allowances before caller buffers
expire and preserves bounded legacy reads. Its rationale and lifetime/resource
tests are in the mean-field-application-budget-transport architecture note. Keep
arithmetic selection separate from SCF AUTO admission: a DF/RSH endpoint may
need strict SCF and still benefit from narrowed derivative work. Strict audit
calculators must never inherit the candidate's private precision request.

Do not simply widen `native_cuda_stationary_binding`'s functional test. Its
controller requires exact full-range derivative bindings and a complete static
source inventory. DF metric response, RSH/VV10 and 3c corrections belong to the
existing phased/composite consumers until equivalent native ownership exists.
The shared generated geometry can serve those consumers without copying their
scientific equations. HF needs its integral/contraction owner, not an XC stub.

The authoritative README runner uses `prepare_batch`, frozen engine-local
post-cold and post-move densities, and real geometry updates. The original
singlepoint-only facade was insufficient; repeatedly constructing singlepoint
calculators is not a valid substitute for README warm timing.
Preserve the concurrently merged resolved-MethodIR and missing-AUTO-force
fallback fixes in `504c023b8`/`124e04d7c` when generalizing this facade.

## Public fitted-force continuation

Bounded public DF PBE/PBE0/B3LYP/r2SCAN/M06-2X now select real typed semilocal
geometry from the budget and publish consumer-owned schema-4 evidence. See
`../implemented/numerics/2026-10-14-public-df-budget-gradient.md` for complete
model/provider gates, strict components, capability/numerical fallback and
independent public endpoint evidence. This supersedes their earlier private-only
boundary, not HF DF or RSH/VV10 work.
Exclusive complete singlepoints show only 1.004–1.109x across the ten measured
dense-grid populations; do not relabel them as all-FP32 or broad 2–7x acceleration.

The public prepared API and actual README Fast runner are now implemented and
qualified separately in
`../implemented/architecture/2026-10-14-prepared-precision-fleet.md`.

## Nonlocal and fitted composite continuation

Dense-grid public direct WB97M-V now has complete independent tests and all 48
native README calls checked against a matching independent GPU reference. See
`../implemented/numerics/2026-10-14-public-wb-budget-gradient.md`; near-neutral
cold and 1.046–1.096x frozen ratios are not broad 2–7x acceleration.

Canonical WB97M-V DF and r2SCAN-3c DF now also select real semilocal FP32 geometry
within their bounded domains. The complete fitted-model checks, 96 native
README calls, unchanged corrections/exact long-range exchange and the retained
losing small moved phase are in
`../implemented/numerics/2026-10-14-fitted-composite-budget.md`.

The overall goal remains incomplete: **HF DF has no admitted mixed operator**,
larger README sizes need qualification and several strict/SCF bottlenecks remain.
The HF note's Slurm 8435 profiles identify different residuals at 24/96/384 AOs;
do not bypass efficient shell schedules just to report float task counts.
Existing direct HF near-neutral gains and the M06 small strict-warm regression
also remain visible. No all-FP32 or all-method/all-size completion follows.

## Master integration and expanded README inventory

PR #2452 was verified merged at `2026-10-14T00:21:24Z`, squash commit
`eeb988b15dccd7bf341e5f293b343b0bcf071ac5`. Follow-up work was preserved in
stash `822c94e695413198d3f27b54f7b5aa2fe7284826` and an ignored full patch/overlay
backup before integrating onto `c2c670e2fa242a479d4792a8464e1998356c2736`.
The old branch and earlier stash remain untouched. The working checkout is
detached at that master revision; no new PR, commit or branch was created.

Retain master's M06-2X AOT names and both packaged spins, its automatic phased
Becke reservation, and its resident/parallel VV10 improvements. The follow-up's
restricted-only package inventory is superseded, not a reason to remove the
upstream UKS package. Fast scientific admission remains independently restricted.
The integrated host suite passed 390 checks. Source-matched GPU build,
requalification and complete timing are separately scheduled. Earlier timings
are historical evidence, not comparisons against these newer optimizations.

The updated README also includes **full-DF CCSD(T)/CBS E+F**, beyond the eight
methods in `benchmarks.readme_dft.METHODS`. The original inventory above is the
HF/DFT matrix, not the whole new README. The user's all-method goal also requires
CBS component methods and complete analytic response, without changing the CBS
formula or using an external production oracle. No correlated/CBS Fast support
is claimed here.

The private DF polynomial continuation and its negative diffuse-metric gate
are recorded in `2026-10-14-df-typed-polynomial-candidate.md`; they do not close
the missing HF-DF endpoint.

The master-integrated complete package (native main SHA256
`3d31a43ddd62630393c6bcd352e6b3333cf9c8004b68258532681e4e488f7ac5`)
passed all **126** scheduled tests in Slurm 8443: 22 complete composite/nonlocal
checks, 58 fitted/prepared/HF/3c checks and 46 direct application-precision
checks. The 48-atom automatic phased-Becke check ran rather than skipping.
Earlier job 8439 failed because a main-library-only build omitted the separately
loaded stationary AOT siblings; building the default complete package in 8441
resolved that prerequisite without changing an accuracy threshold. Job 8444
runs the source-matched complete README timing separately.

This package predates the efficient private DF shell/Rys candidate and its
compensated FP32 accumulation. That candidate's implementation, retained failed
gate, replay semantics and independent native/sanitizer evidence are in
`../implemented/numerics/2026-10-14-df-shell-typed-runtime.md`. Do not relabel
the 126 endpoint passes as public HF-DF FP32 qualification or its speed evidence.

## Completed integrated timing and native HF-DF follow-up

Slurm 8444 completed all 24 engine/population runs and 288 gated calls. The
eight matched populations have positive FP32 geometry work, one solve and the
same SCF iteration/Fock counts in strict and Fast. The new evidence, including
the small-3c moved loss, is in
`../implemented/performance/2026-10-14-master-fast-readme-requalification.md`.
Cold and frozen timings remain distinct; the M06-2X frozen gains are not cold
singlepoint gains and do not qualify all README sizes.

The previously missing HF-DF arithmetic now has a **bounded numerically
qualified native shell candidate**, superseding the private-only boundary.
See `../implemented/numerics/2026-10-14-native-hf-df-budget-gradient.md` for exact
final-density ownership, rank/shape gates and negative performance evidence.
The first actual 96-AO README timing loses roughly five percent; the 24-AO
generic path still performs no mixed work and pays its strict audit fallback.
Do not mark either result as useful acceleration. The follow-up deferred-vote
guard change passed native/sanitizer and complete numerical tests but lost
timing more severely in job 3003 and is reverted. Jobs 3004/3005 build the next
cost-selected candidate with early rejection restored and s/p-only orbital bra
classes retained in FP64. It passes 57 selected checks in job 3006 and 72 gated
README calls in job 3007, but the 96-AO endpoint is only near-neutral
(1.003–1.006), not useful demonstrated acceleration. The 24-AO generic route
still has no mixed work and pays two strict audit solves.
Larger sizes, generic/SCF narrowing and full-DF
CCSD(T)/CBS remain open, so the overall goal remains active.

## Larger fitted-domain qualification

`test_large_df_fast_candidate_cuda.py` now exercises the private semilocal
consumer at 384 orbital AOs / 48 atoms on the existing WATER27-derived synthetic
pair of octamers. Five fitted local models use the dense 48/16/32 grid and an
independent GPU4PySCF metric/SCF/analytic-gradient oracle at both geometries.
Strict and mixed prepared owners start independently cold, retain their own
frozen seeds, move geometry, and return to strict arithmetic. The unchanged
gates are 1e-7 Eh energy, 1e-5 maximum / 3e-6 RMS mixed force, 1e-6 strict force
and 2e-8 translational force sum. Public admission is deliberately unchanged
while this candidate qualification runs. Tests are not clean timing evidence.

Initial job 8453 passed every strict/mixed numerical phase for PBE and PBE0,
then exposed a test-harness error on the final strict return: omitting the
coordinate argument requests the prepared original geometry, not the moved
geometry. The test compared that result with a moved force. After stopping
the remaining tests, the fixed test explicitly replays the moved coordinates;
all input geometries and error gates are unchanged. Preserve the failed log/XML.
Job 8454 passed the full five-model candidate population in 953.62 seconds,
without skips. This covers independent cold/frozen/moved/strict-return checks
for PBE, PBE0, B3LYP, r2SCAN and M06-2X at 384 AOs and 48 atoms. Library SHA256
is `3d31a43ddd62630393c6bcd352e6b3333cf9c8004b68258532681e4e488f7ac5`;
the source matches the master-integrated DFT package, not the newer native HF-DF
shell build. Public admission remains unchanged: these private numerical
checks do not establish complete public-route qualification or clean timing.

At the user's explicit request to wrap up, no further domain expansion or
correlated implementation is started. The all-method/all-size goal remains
unfinished rather than being relabeled complete. Stage PR #2452 was freshly
verified merged; the follow-up remains an uncommitted detached worktree on
`c2c670e2fa242a479d4792a8464e1998356c2736`, with ignored recovery patch/overlay.
The final bounded provenance-cache fix is separately recorded in
`../implemented/performance/2026-10-14-fast-model-provenance-cache.md`:
job 3009 passes 74 GPU checks with one missing-dftd4 skip, and job 3010 gates
all 72 actual HF-DF README calls. At 96 AOs, frozen gain is about 1.048x;
cold remains neutral and the 24-AO strict-audit loss remains visible. No further
FP32 coverage is inferred from this host-bookkeeping improvement.

## Correlated/CBS ownership discovered on integrated master

`CCSDTCBSCalculator` currently creates grouped `Calculator(precision="fp64")`
owners and rejects precision options in its common-options whitelist. The
application-budget transport is also explicitly restricted to CUDA HF/KS in
both Python and native registry admission. Adding a Fast spelling to CBS alone
therefore cannot authorize a correlated FP32 consumer.

Importantly, ordinary DF-CCSD already requests bounded FP32 auxiliary warmup
for `nocc >= 9 && nvir >= 128` via `default_df_ccsd_mixed_precision`, even under
the current FP64 public endpoint contract; final amplitude/residual replay is
strict. This existing default belongs in the optimized comparator. Do not
disable it in the baseline, or count it as new application-budget Fast work.
Lambda's separate internal mixed option is still opt-in. Triples expose their
own precision directive and diagnostics rather than the HF/KS gradient census.

A complete CBS extension must preserve same-basis shared references and the
linear-combination coefficients, allocate accuracy across the actual grouped
components (including coefficient amplification), and qualify final analytic
response and the composite observable. Correlation-only historical mixed-warmup
evidence does not qualify the README's full-DF HF/RI-JK Hamiltonian. No CBS
admission, new arithmetic or acceleration is implemented by this investigation.
