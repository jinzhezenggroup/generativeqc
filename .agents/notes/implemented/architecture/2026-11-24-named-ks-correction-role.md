# Decision: bind the named KS correction role to native discovery

Status: implemented
Date: 2026-11-24

## Problem

The constructor accepted the native named identity but reclassified the graph
to select a facade carrier, decide whether to detach D3/D4, and reject unsupported
correction shapes during budget preflight. This duplicated native catalog knowledge.

## Decision

Publish role and carrier with the backend-bound selection, separately from the
carrier used by native prepare. Default named construction consumes this metadata;
explicit compositions and older catalogs keep their original graph-based route.
The graph is still verified and used for public inspection, electronic projection
and remaining frontend checks. An incomplete or invalid current wire is final.

## Invariants

A successful named selection is not contextual admission. Detached means exactly
one represented D3/D4 term; do not use it to add a force or Hessian capability.
The pinned intrinsic PBE-D4 row and basis-bound r2SCAN-3c do not use that budget
shape. The native prepared owner still executes all terms and publishes results.

## Revisit when

Move the remaining constructor consumers to typed native metadata before making
the `MethodIR` view on demand. Compare complete effects against both the old
catalog route and independent numerical oracles before retiring it.

Refs #2227.
