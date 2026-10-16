# Decision: share basis capacity with general-template expansion preflight

Status: implemented core foundation; versioned facade consolidation is pending
Date: 2026-10-16

## Problem

The independently reviewed #2537/#2539 and #2547 branches introduce overlapping
catalog/API ownership. The former routes both compiled named records and custom
general-contraction/scalar-ECP templates through one expansion owner. The latter
adds useful byte-counted discovery, provenance and detached-buffer admission,
but its separately generated flat catalog and named expansion loop are not a
second independent scientific increment that can simply be landed alongside it.

## Decision

Use the general-template implementation as the one atom/contraction/ECP expansion
owner. Before adapting the versioned catalog facade, extract its existing
dimension preflight into one checked, borrowed-data routine shared by the shape
query and actual expansion. Named capacity and named expansion construct the
same borrowed compiled template; neither adds a second table or atom-expansion
implementation. Discovery names and the checksum borrow the existing pack.

The checked template retains its fixed-size element lookup on the stack. It
validates all descriptor ranges and coefficients, then counts every repeated
atom and contraction row before allocating output vectors. Also bound expanded
AO width, not just shell/primitive indices: valid individual angular momenta can
otherwise exceed native AO dimensions despite individually representable raw
buffer counts. On 32-bit hosts, check both byte products and their sum before
publishing detached-buffer capacity.

This branch does not yet deliver the #2547 installed versioned C facade or its
Python cutover on the shared owner. Those adapters, public contracts, allocation
failure/alias controls, exact-source integration tests and complete numerical
endpoints remain required before the catalog consolidation is review-ready.
The existing PRs and their recorded qualification heads are not rewritten by
this local foundation.

## Invariants

- Preserve atom, shell, general-row and coefficient order, including zeros.
- Shape queries inspect element identities and representation, not coordinates,
  charge, multiplicity, normalization or scientific/backend admission.
- `buffer_bytes` covers detached orbital shell/primitive buffers only. Per-atom
  core counts, scalar-ECP terms, normalized system copies, allocator overhead,
  retained workspaces and providers require separate accounting.
- Scalar-ECP term count remains separately visible; existing ECP channel mapping
  and normalization/scientific owners do not move into the capacity query.
- A refused native request is not permission to reconstruct scientific data in
  Python or consult an alternative catalog owner.
- Memory-bounded counting is not constant work: complete descriptor validation
  and expansion counts still scale with supplied records and atom occurrences.

## Rejected alternatives

Counting the output of an actual expansion spends the very numeric allocation
that admission must precede. Keeping #2547's flattened generator and numerical
loop would preserve two production owners after an ostensibly shared migration.
Using Python shell counts as authority would leave the installed SDK/CLI with a
different resource model. The byte-counted facade should instead adapt the
shared native preflight while retaining its own ABI alias/extension checks.

## Validation boundary

Native controls cover the known general-row/ECP fixture, repeated d shells in
both representations, shape-only handling of non-finite coordinates, and AO
overflow rejected before expanded-vector allocation. Catalog controls compare
all compiled records in both representations with actual shared expansion,
including borrowed discovery aliases and identical pack provenance. Full
canonical-record/PySCF parity and the versioned facade's negative controls remain
integration gates, not inferred passes from these core tests.

References: #2227 R2/R5/R6, #2537, #2539, #2547; the earlier shared bundled-basis
and element-template Agent Notes preserve the original decisions.
