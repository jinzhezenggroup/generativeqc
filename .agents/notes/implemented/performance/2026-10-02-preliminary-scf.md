# Decision: explicit bounded preliminary SCF before the immutable target

Status: implemented; CPU native and host qualification passed, CI pending
Date: 2026-10-02

## Problem

A preliminary calculation can reduce target SCF iterations while increasing
complete endpoint time. Under #1246, the existing xTB charge/orbital prototypes
already demonstrate the mechanism, but do not establish automatic profitability.
An explicit HF/LDA preparation also needs reliable bounds, diagnostics, and
compatibility with existing warm-state ownership.

## Decision

Add an opt-in execution policy with the existing RHF or LDA provider. The first
scope is CPU FP64, restricted all-electron exact energy endpoints. The target's
method/basis/grid/precision/tolerances remain immutable, and each target attempt
constructs fresh iterative state. Existing explicit/imported/retained density is
authoritative. The preliminary source is transient; its density is validated in
the target AO metric before use. Preparation runs once and a failed seeded
attempt gets at most one ordinary core retry. Allocation errors propagate.

The two-solve orchestrator lives beside the upper-level SCF drivers in
`src/scf/preliminary_guess.*`. The lower `src/scf/initial_guess/` layer retains
only pure policy types and its existing density algebra/validation; it does not
acquire HF/DFT method-driver dependencies. No scientific equation is duplicated,
and no policy is inserted into the generic compiler. The provider consumes
existing typed Fock/semilocal execution contracts; it does not branch on a target
functional's display name.

Warm-state storage, checkpoint identity, update/freeze and last-good publication
are unchanged. In the active opt-in mode, a failed warm attempt goes directly to
core, cannot reactivate preliminary SCF, and cannot retry an allocation error or
start a third target attempt. Default/no-policy retry behavior is unchanged. In particular, this policy does
not implement the shared MP2/RCCSD/RCCSD(T) warm-reference lifecycle already
published in #1696 and being repaired separately.

The native cap covers additional numeric payloads before allocation. The public
resource planner composes the existing full target/preparation host inventories
conservatively, including overlap, so global host budgets cannot hide the new
owner. The narrow numeric cap is not mislabeled as process RSS. Scoped retained
capacity observations include the idle target during preparation and the
proposed seed during the target attempt.

## Rejected alternatives

- Unconditional or size-only default promotion: current measurements are too
  narrow, and multiple stationary solutions require domain-specific checks
- Tight/full-grid preliminary DFT: it reduced iterations but lost end to end
  in the measured 6-31G cases
- Moving the initial-guess algebra layer upward into a method dispatcher:
  orchestration belongs with drivers; metric validation stays independent
- Reusing DIIS, pretending xTB is an all-electron target basis, or altering
  target controls: these would change the scientific/ownership contract
- Silently running the preparation on CPU for a CUDA target: not admitted
- Duplicating or changing the unrelated correlated-method warm-state lifecycle

## Evidence

The exploratory bridge used the clean audited source tree
`42cc3f8f871523a4d48304f520a5e967e61cea6e` at local revision
`8ba615aacd9f02708b39782cc71b68f27bbe1af6`; its relevant SCF/DFT source files match
master `e4bd02570a9063dba6e107448826c70714d55c75`. The CPU library SHA-256 was
`670c81365bde477f32e5a7fc6af1cff5e489a42e118651aca77090fd00626258`.

On a 12-atom/52-AO water tetramer with the same PBE0/6-31G target and three fresh
process pairs, direct median total cost was 44.4100 s. HF preparation reduced
target iterations 18→11 and total time to 31.2357 s, including 1.7148 s
preparation. Coarse LDA also gave 11 target iterations and 31.9636 s total,
including 2.6238 s preparation. The small HF/LDA timing difference is not a
robust provider-ranking claim. Maximum paired final-energy differences were
below 2.1e-12 Eh and density differences below 7.2e-10.

Conversely, same-grid PBE preparation for water/6-31G reduced target iterations
12→8 but raised total cost from 0.4572 to 0.7380 s. This negative result is why
iteration counts alone cannot authorize selection.

These are energy-only CPU exploratory measurements, not a production-default,
large-organic, open-shell, ECP, transition-metal or GPU qualification. Preparation
uses a separate integral owner; integral sharing between stages was not tested.
The independent water/PBE0 same-grid PySCF spot check differed by 1.71e-13 Eh.
The actual feature library passed seven selected native suites, including the
new provider/failure/warm-priority contracts, and all new Python API/controller
cases. The energy/fallback matrix includes RHF, PBE, PBE0, r2SCAN and B3LYP
targets with both HF and LDA preparation. An independent reviewer also ran
67 existing default-path CPU regressions with zero skips. The production
feature remains disabled by default regardless of these observations.

## Consequences and revisit conditions

Users can choose a provider explicitly and inspect all preparation/fallback
work. A global resource plan is deliberately conservative and may reject a
budget that would fit a narrower optimized lifetime schedule. Broader method,
basis, electronic-state, atom-count, backend and failure-rate evidence, with
complete endpoint cost and state checks, is required before an automatic
profitability policy or a new default is proposed.

## References

- #1246: xTB-informed initial guesses and complete endpoint acceptance
- #1247, #1250, #1289, #1294: existing export, mapping and provider foundations
- `docs/user/initial_guesses.md`: current API and admitted scope
- `tests/native/test_preliminary_initial_guess.cpp`
- `tests/python/test_preliminary_initial_guess.py`

## Three-stage counterexample

The follow-up used four fresh-process repeats per route in Latin-square order.
For the same 12-atom/52-AO target, direct PBE0 was 44.9289 s median;
HF→PBE0 30.3529 s; coarse-LDA→PBE0 31.9578 s; and
HF→coarse-LDA→PBE0 32.6765 s. Three stages reduced the intermediate LDA solve
from 17 to 11 iterations, but all three candidate routes required 11 target
iterations. Its 3.8782 s preparation did not beat the single-HF route's
1.6027 s preparation. All 32 comparison endpoints matched the same restricted
solution, with maximum energy/density deltas below 2.1e-12/7.2e-10.
The four-repeat small-water comparison also showed no added target-iteration
benefit. The chain remains experimental and is not added to the API.

## Actual public-API qualification

The separately retained `benchmarks/results/preliminary-scf-api/` campaign runs
18 new `Calculator` endpoints in three balanced fresh-process repeats. On the
12-atom/52-AO tetramer, constructor + preparation + execution medians were
43.6234 s direct, 30.6409 s with HF, and 31.4944 s with coarse LDA. All target
states matched within 2.05e-12 Eh and 7.13e-10 maximum AO-density difference.
The complete child-process medians were 44.1790, 31.1020 and 31.9693 s.
The process RSS high-water medians rose from 327,704 KiB to 388,016 KiB;
that includes imports and state export, not just the preliminary owner.
Small-water HF improved the endpoint median but lost on whole-process median.
These costs and negative observations remain part of the evidence; three
repeats and native paired checks do not satisfy the default-promotion gate.

After merging frozen master `ac6080759`, the incremental CPU build changed
only build-identity compilation (plus regenerated DF contract metadata) and
relinked the library. Seven native suites, 62 new-API Python cases and 67
existing default-path regressions passed again. The performance source/binary
identities remain pinned to the earlier measured campaign; these integration
checks are not relabeled as a new timing campaign.

## Review corrections

Follow-up review aligned resource planning with the native early LDA decline
for g-shell targets: the target inventory is retained, while only eligible
items contribute preliminary reservations in mixed batches. Tests include a
charged eligible item following an ineligible one and preserve genuine provider
errors. The public initial-guess tag follows the header's fixed-width int32_t
convention, with normal and `-fshort-enums` clients tested against the same
native parser. These are untimed follow-up corrections; the historical timing
source and binary identities remain unchanged.
