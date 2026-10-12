# Decision: reuse compact screened layouts for explicit partial ERI sources

Status: implemented (experimental controls; no partial-default promotion)
Date: 2026-10-11

## Problem

The first partial-source experiment deliberately refused positive screening.
After master acquired compact screened global-hybrid sources, that refusal
prevented testing PBE0 under its real production Schwarz predicate. Using an
unscreened PBE0 ablation or disabling master's existing full cache would not
answer whether partial residency improves the ordinary complete endpoint.

## Decision

The common provider uses its existing immutable resident bucket counts for
selected-byte inventory and construction, including the exact same compact
Schwarz rows, bound predicate and launch work extents as complete residency.
Unselected buckets remain in the common compensated canonical Direct action;
no second full-source pass or duplicate recurrence owner is introduced.

KS accepts separate explicit `GENERATIVEQC_KS_RESIDENT_BUCKET_MASK` and
`GENERATIVEQC_KS_RESIDENT_MAX_BYTES` controls. A common strict unsigned parser
also serves the existing RHF controls. Unset still means every bucket and the
incumbent method-specific ceiling. Existing automatic eligibility, native-cold
provenance, resource-ledger refusal and warm handoff remain unchanged. The
default-promotion inventory now registers all four experimental controls as
needing qualification and protects the full-mask default against mutation.

At the frozen threshold, the lease audits exactly the source candidate count
and exactly the uncached recurrence count. A stricter replay threshold still
visits every frozen row but can legitimately skip Direct evaluations. Its
evaluation count is bounded above by the uncached inventory rather than
incorrectly requiring evaluation of screened-out quartets. Lower replay
thresholds remain invalid. This distinction affects audits, not screening math.

## Why not an unconditional partial default

The prior constrained RHF improvement does not establish a PBE0 or full-cache
crossover. Master already admits screened PBE0 full sources automatically.
Partial storage can save setup/memory but must recompute uncached buckets;
replacing an admitted profitable full source can therefore regress. A future
selector must distinguish an admitted full cache from a natural capacity
refusal, charge construction and complete consumer work, and qualify reuse
and resource crossovers rather than hard-code the measured water geometry.

## Validation and evidence boundaries

Host tests exercise compact selected inventory/construction, immutable masks,
source retirement and refusal, strict unsigned syntax, exact frozen-threshold
census and stricter-threshold evaluation bounds. Device probes compare signed
raw J/K against independent PySCF for both screened and unscreened full/partial
leases, with PBE0/B3LYP coefficients and same-grid energies.

The observation-only `benchmarks/selective_eri_pbe0.py` delegates the complete
Cold/Warm/Moved/Moved-Warm endpoint and independent E+F gates to the unchanged
official PBE0 benchmark. It retains strict precision, source preparation,
handoff and semantic work receipts without changing seeds, density, XC or force
ownership. New PBE0 measurements pin master
`bd6f78288d466c254a372f6e39a0d0a31139a479`; the earlier RHF bundle remains pinned
to its original measured baseline and is not relabeled as newer evidence.

Endpoint outcomes are recorded only after the finite Slurm jobs complete.
No performance result, sanitizer pass or broad/default qualification is implied
by implementing the experimental controls or by launching a job.

### Actual-master PBE0 negative crossover

Job 7538 completes six independent signed raw-J/K and same-grid PBE0/B3LYP
GPU tests, then four fresh complete endpoint processes in ABBA order. Job 7545
completes full/partial, screened/unscreened f-shell memcheck with zero errors.
The host policy, lifetime, ownership, evidence and benchmark suites pass 527
tests. The build pins Release/sm120/CUDA 12.9.1 and reuses ccache.

For 96-atom water/def2-SVP, production screening 1e-12 and the official moving
grid, master's **natural unset full cache** takes median 75.560 s Cold;
explicit mask 32752 takes 112.395 s, a 48.75% regression. Four-phase sequence
medians regress 22.39%. Cached ERI bytes fall from 14,985,773,728 to
11,268,906,016 (24.80%); this is value storage, not a profiled total VRAM peak.
Each arm performs the same 18/1/12/1 physical Focks. The cold source serves 17
actions, followed by one separately charged incumbent handoff. Partial saves
construction/storage but adds 464,608,464 Direct recurrence evaluations per
action. Full-source auditing requires zero replay evaluations; stock master
does not emit the newer detailed canonical census trace fields.

All 16 complete E+F gates pass: maximum independent energy/force errors are
9.55e-12 Eh / 3.79e-11 Eh/Bohr; paired errors are 1.82e-12 / 1.63e-12.
This rejects **replacing admitted full PBE0 storage with this fixed partial
mask**. It does not reject every selective strategy or establish behavior
under natural full-cache refusal. The former constrained RHF improvement is
still a separate scope; neither campaign supplies an automatic profit model.

Retain full-cache defaults and explicit experimental controls. Two exploratory
pairs are not formal performance/production/default promotion. The compact
negative summary in `benchmarks/results/selective-eri-pbe0-crossover-20261011/`
retains timings, source/work identities, gates and raw hashes. Full vectors,
traces, inputs, build logs and sanitizer receipts remain in ignored local
artifacts and the finite-job fixture on n1; no release/assets or external
backup is published. This negative summary is not a complete promotion bundle.

## References

- [Original partial-source decision](2026-10-11-partial-canonical-eri-residency.md)
- [Current SCF ownership contract](../../../../docs/developer/scf_module_boundaries.md)
- [Existing PBE0 full-source default evidence](../../../../benchmarks/results/hybrid-phase-source-default-20261011/README.md)
