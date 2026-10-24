# Decision: account for physical layout transfers before joint selection

Status: implemented
Date: 2026-10-20

## Problem

Issue #1889 requires precision/provider/layout/fusion choices to include packing
traffic, complete phase cost and simultaneous storage. On master 457a4795 the
logical-axis check admitted a candidate that changed the first matrix's strides
from (3, 1) to (1, 2), with zero packing traffic and zero temporary storage. Its
one-unit kernel estimate beat the unchanged layout. A different layout hash
cannot establish that the original bytes were copied into that layout.

## Decision

Represent a bounded same-storage floating affine copy as LayoutTransfer, binding
the exact source/destination OperandLayout, before/after phase and provider-owned
copy implementation. Require a pre-copy for changed read views and a publication
copy for changed write views. Read/write views need both, with one intermediate
buffer charged once. Copy traffic counts logical visits; padded storage charges
the full affine address span. These are conservative logical bounds, not measured
DRAM traffic. Copies may be fused into the kernel cost, but their traffic and
capacity obligations remain explicit.

Joint collection/selection retains incomplete transfer offers as rejected
evidence, including when the request has no optional resource constraints.
Qualified-incumbent retention cannot bypass these checks. Additional storage
continues through the existing resource sum and exact request budget.

The existing native portfolio ABI has no generic copy executor. Emit an explicit
unsupported reason for conversion recipes rather than serializing only their
layout hash and silently discarding their execution obligation. This changes no
runtime ABI or admitted native numerical implementation. A future runtime owner
must implement, resource, capture-qualify and independently validate the copy
sequence before this boundary can admit it.

Empty transfer lists are omitted from payloads, preserving old generated native
portfolios and candidate identities. Physical views may demand weaker alignment
without a copy; stronger alignment or different strides require explicit copying.

## Rejected alternatives

- Treat a changed stride/stronger alignment as a zero-cost pointer reinterpretation.
- Add estimated pack bytes without an explicit bound copy/publication obligation.
- Advertise native execution because abstract compiler selection can rank a recipe.
- Infer alias, triangular, virtual-expression or cast-plus-pack legality here.
  Those need separately qualified lowerings; this slice retains rejection.

## Evidence

The reproducer and baseline identity comparison live under
.artifacts/1889-layout-transfers-20261020/ in the single Inspire checkout.
test_layout_transfer_binding.py covers independent scalar product parity after
a real NumPy affine copy, untouched padding, complete-vs-kernel-only cost,
exact padded-capacity limits, read/write/publication obligations, stale
endpoints/dtype, unsupported ownership and actual compiled native fallback.
Existing joint/provider, native, DFT and CC suites remain the compatibility gates.
No new GPU route, mixed-precision science or endpoint speedup is qualified.

## Revisit when

A shared runtime/code-generation owner can execute and account for the complete
copy/contract/publication sequence, or separately establishes alias, triangular,
virtual materialization or fused storage-cast legality. Reuse this shared
contract instead of another method-local packing flag.
