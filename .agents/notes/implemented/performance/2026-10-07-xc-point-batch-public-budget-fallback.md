# Decision: Keep budgeted KS point batching on its incumbent

Status: implemented
Date: 2026-10-07

## Problem

The public KS resource inventory reserves incumbent XC storage, the retained
owner fleet, setup excess and later force capacity. It does not reserve optional
point-batch panels. Default panel allocation under that shared numeric ledger
can succeed for early owners by spending capacity needed by later mandatory
owners or workspace. Rejecting a later allocation cannot reclaim earlier panels.

At source `8dda81a75ed5b61192a79981c20ffb725f207c07` (unchanged in evidence-only
head `9b456f0637f2b7907e34ccee47ea1174eed43526`), the real public plan for 1,024
H2/STO-3G PBE owners, tile 256 and grid 48×16×32 creates a 2,199,167,612-byte
ledger. Optional panels occupy 1,245,184 bytes per owner. A host allocation-double
replay of native shapes and the real ledger stops after 674–675 owners, while
explicit opt-out prepares all 1,024.

## Decision

The shared native KS preparation boundary forwards a zero point-panel allowance
whenever a native device ledger is active. This preserves the existing one-tile
fallback without increasing the public estimate or user budget. It covers single
owners, retained fleets and geometry rebuilds through the same native path.

An explicit `ResourceBudget()` with unlimited user caps still creates a numeric
ledger from the incumbent-only inventory, so it also retains one-tile execution.
Batch environment overrides cannot bypass that boundary. Unbudgeted ordinary KS
keeps the bounded default. The lower-level XC owner still accepts its caller's
explicit additional allowance; its existing allocation-rejection fallback stays
unchanged. CPU policy, scientific point algebra and force tiling are unchanged.

## Rejected alternatives

- Counting current free ledger bytes as an optional allowance cannot protect
  future owners or later retained force storage
- Making panels universally mandatory in the public estimate would reject
  budgets that fit the incumbent instead of taking the bounded fallback
- Delaying panels until fleet setup alone does not protect later force storage
- A separate optional-allowance ABI and planner candidates may be useful later,
  but are unnecessary for this narrow correctness repair

## Evidence and limits

`tests/python/test_ks_point_batch_resources.py` compiles native inventory queries,
KS point-panel policy, the compiler-emitted residency planner and the common
allocation ledger with host CUDA allocation doubles. Both unlimited and exact
user caps prepare the 1,024-owner fleet and leave a later 128-MiB workspace intact;
a single owner leaves the full 512-MiB staging allowance intact. Removing only
the new guard reproduces the fleet failure and the single-owner workspace OOM.
Unbudgeted default allocation and all existing explicit opt-outs are tested.

These tests certify policy/accounting behavior, not GPU endpoint execution or
performance. They do not claim the pre-existing inventory is complete: its
legacy XC slot includes 32 bytes per grid point while the resident grid retains
40. The fixture includes that known extra residency and uses a bounded later
workspace that still fits the original plan; the separate legacy gap is not
silently repaired here.

## Revisit when

A public plan explicitly reserves optional panel lifetimes separately from all
mandatory owner/force capacity and conveys the selected allowance to native
preparation. Any such extension must preserve incumbent-fitting budget admission.

## References

- PR #2089, review 5444452614
- `2026-10-07-xc-point-batch-default.md` records the original default promotion
- `docs/developer/xc_native_cuda.md`
