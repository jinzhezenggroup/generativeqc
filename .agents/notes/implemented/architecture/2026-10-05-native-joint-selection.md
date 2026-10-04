# Decision: bounded native projection of joint lowering

Status: implemented
Date: 2026-10-05

## Problem

Native preparation cannot call the Python lowering selector. Reimplementing
provider or precision policy in each method would lose the common request,
negative evidence, complete phase costs and strict fallback semantics of #1889.
Existing deployed implementations also lack complete phase timing; inventing
zero costs would turn migration into an unsupported performance promotion.

## Decision

`common.native_lowering` projects the canonical request/candidate portfolio into
static C++ records. `runtime/lowering_binding.hpp` applies the same selection
rules at preparation: at most 256 candidates and 16 precision variants,
context/capability/resource/determinism/capture gates, complete preparation plus
expected replay cost, and lexical identity tie breaking. The native adapter
must resolve executable lifetimes and runtime shape compatibility separately.
Template digests do not certify arbitrary runtime dimensions.

Both selectors also accept an explicitly identified, already-qualified incumbent.
When that incumbent lacks complete cost evidence they preserve it after all
legality gates, retain the missing-cost evidence, and require an executable
strict requested-precision candidate. An estimate for a competitor cannot beat
an unknown incumbent. If the incumbent has complete cost evidence, ordinary
cost ranking applies. This option preserves a deployed implementation; it is
not scientific qualification or a performance-default promotion mechanism.

## Invariants

- Alternative arithmetic requires scientific qualification. Publication and
  input arity cannot change. No implicit TF32 or hidden publication cast.
- Unknown preparation, conversion, packing, refinement, audit or fallback time
  remains unknown. Selection diagnostics distinguish incumbent retention.
- Fallbacks retain selected arithmetic or restore strict requested precision.
- Sum simultaneous workspace, provider, temporary and cache ownership; reject
  invalid identities, duplicates, bounds and integer overflow before execution.
- No GPU work, library heuristics, device allocation or per-iteration search is
  part of this selector. Its bounded host vectors belong to preparation.

## Evidence and limits

Cross-language tests compile the emitted metadata and C++ selector with ccache,
then compare selection, fallbacks and rejection counts to Python. They cover
amortized preparation, unknown costs, stale targets, precision/capability and
resource rejection, exact-order capture, incumbent retention and overflow.
The relevant contract suite passes 33 tests. This slice introduces no production
consumer and makes no GPU numerical or endpoint performance claim. DFT and CC
binding migrations and complete endpoint qualification remain work for #1889.

## Rejected alternatives

Treating unmeasured phases as zero, guessing hardware timing from FLOP counts,
or using per-method provider booleans would hide missing evidence. Requiring
complete timing simply to preserve a deployed implementation would block the
ownership migration without improving numerical safety.

## Revisit when

Native consumers can supply complete target-specific phase evidence and typed
prepared executions for all competing candidates. Such evidence may enable
cost ranking; it must not change the scientific identity or admission boundary.
