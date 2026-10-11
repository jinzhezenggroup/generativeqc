# Decision: opt-in screened cold Global-Hybrid auto candidates

Status: implemented
Date: 2026-10-11

The opt-in default is superseded by the bounded
[default-auto decision](2026-10-11-hybrid-phase-source-default.md).
The measurements and opt-in boundary recorded here remain historical.

The warm-baseline handoff is subsequently corrected by the
[provider-consistent warm-energy decision](2026-10-11-hybrid-warm-energy-handoff.md).
The job 7242 measurements below remain unchanged pre-fix evidence; auto
eligibility, capacity admission and the opt-in default policy are unchanged.

## Problem

The compact-source implementation admitted the 96-atom PBE0/def2-SVP cold
endpoint only under forced selection. Its SPD domain and 14,985,773,728 value
bytes are outside the inherited generic-f/8-GiB auto gates. Simply relabelling
forced selection as auto would also attempt setup on seeded and warm solves.

This supersedes only the KS auto eligibility/cap decision in
[the screened source note](2026-10-10-screened-hybrid-phase-source.md).
That note's forced E+F measurements remain historical forced evidence, not
auto qualification.

## Decision

Keep unset KS selection disabled. Add an explicit method-neutral auto policy
parameter whose default preserves RHF's qualified generic-f domain and 8-GiB
cap. KS explicitly chooses screened-cold candidates: through-d sources need a
positive immutable Schwarz preparation mask and may use at most 16 GiB of
values. Generic-f auto retains its prior capability gate and 8-GiB ceiling.
No molecule-name/atom-count branch, density seed, approximation or FP64 change
is introduced. All complete numeric/live-device admission and bounded exact
fallbacks remain mandatory, including the unchanged 256-MiB runtime reserve.

Reject auto warm/seeded and below-threshold candidates before source cloning
or row sorting, with an observable host-decision trace. The lease repeats the
policy check for callers outside KS. An unseeded changed geometry is eligible
as another cold phase; a solve using a moved warm snapshot is not. Forced
selection remains a bounded experimental override, not an automatic policy.

The first actual-auto endpoint (job 7237) correctly failed its positive-source
observer: the high-level default native guess arrives at CUDA KS as a non-null
density, so the original pointer-only cold test refused it as a reused bucket.
Retain this failed evidence. Pass an explicit `CudaKsSeedOrigin` from the native
guess owner, both single-system and batched, only when its own cold preparation
produced the seed. Defaults and warm projections remain External; seed math and
normalization do not change. Do not qualify forced-source runs as a substitute
for exercising this production auto scheduling boundary.

The existing 64-public/128-Cartesian-AO and eight-permitted-iteration gates are
candidate eligibility, not a calibrated cost model. Actual compact counts
decide memory feasibility. They do not predict convergence, recurrence versus
scatter cost, or source amortization. Adding an unmeasured minimum byte count
would not produce such a prediction either. Default promotion remains blocked
on complete repeated E+F/work-count evidence, small/generated cases, resource
cliffs, and a complete public resource-planner allowance. A faster cold sample
cannot compensate silently for warm/whole-sequence regressions.

## Validation

Host policy/lifetime and inventory mutation gates cover method-policy isolation,
cold/warm, masks, angular coverage, AO/iteration limits, caps, budget and skipped
setup. Actual-auto device qualification and source-matched endpoint evidence
are retained separately from the frozen forced-source artifacts under
`.artifacts/hybrid-phase-auto-20261011/`. Qualification results are appended
after execution; no timing result is assumed by this decision.

### Source-matched actual-auto endpoint

Slurm job 7242 uses embedded/uploaded source identity
`90044c5174f739d2394867908bbc42e7f741aed4cf420e4ec1465235809047d1`.
Four fresh processes execute `0, auto, auto, 0`, each with the unchanged official
96-atom cold/warm/moved/moved-warm PBE0 complete E+F protocol and one fixed-density
replay per geometry. All 16 independent gates pass, with maximum energy/force
errors 9.55e-12 Eh / 3.78e-11 Eh/Bohr. Both auto cold phases really construct
1,873,221,716 values (14,985,773,728 bytes) and complete 17 audited Fock actions.
Later auto phases skip source preparation entirely, not capacity-fallback setup.

| Phase | Disabled median (s) | Auto median (s) | Disabled / auto |
|---|---:|---:|---:|
| Cold, including preparation | 91.078 | 75.976 | 1.199 |
| Warm | 16.226 | 25.578 | 0.634 |
| Moved | 64.329 | 66.107 | 0.973 |
| Moved-warm | 16.295 | 16.227 | 1.004 |
| Complete four-phase sequence | 187.928 | 183.889 | 1.022 |

Disabled cold Fock counts vary 19/17 versus 17/17 in auto. Warm counts are 1/1
versus 3/4. Eliminating warm source setup does not eliminate the different
cold trajectory's warm-snapshot work. One ABBA block is not a confidence
interval, and the small whole-sequence ratio does not justify default
promotion. Do not describe these ratios as isolated or fixed-work J/K wins.

The same job's 3-atom generated case passes eight independent E+F gates; auto
skips source setup for every phase. Final-library job 7243 passes four f/SPD
water probes (both PBE0/B3LYP at chunk widths 1/2), including signed raw J/K,
independent same-grid energies, native-cold versus external/warm provenance,
numeric-budget refusal and public-ledger fallback. Host regression is 480
passed/12 GPU-skipped; four unrelated GFN1-D3 asset-dependent checks are
excluded after recording their missing offline-source prerequisite.

The full 24-atom SPD memcheck is a separate finite scheduled qualification,
not a prerequisite silently treated as passed by the endpoint observer. The
earlier pre-provenance-fix job 7236 passed six normal GPU probes but its
memcheck hit the 15-minute scheduler limit. A trap's exit_code 0 does not
override scheduler cancellation or replace an explicit sanitizer summary.
Retain raw outcomes and require positive completion for sanitizer claims.

Final-library job 7243 subsequently completes the full 24-atom SPD probe under
its 30-minute Slurm limit, with both functional receipts and `ERROR SUMMARY:
0 errors`. An additional finite 10-minute water/f-shell memcheck (job 7248)
also completes with both receipts and zero errors. Both have explicit final
completion markers; the earlier timed-out scanner is not substituted for them.
