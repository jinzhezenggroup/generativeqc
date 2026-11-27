# Decision: resolve the default named CPU KS grid in the catalog

Status: implemented
Date: 2026-11-24

## Problem

The Python facade already obtained the exact default graph from the native
catalog, but still used the compiler grid policy and a second inspection of
the graph to decide whether the VV10 grid should share the primary grid.

## Decision

Project the omitted XC and VV10 grid levels from the same native selector
and exact identity, without constructing a context. Read the SCF tail/spin
domain and nonlocal-presence flag from that same electronic program instead
of lowering the graph again just to validate the option wire and memory cap.
The facade still owns
representation of a grid as `GridSpec`, honors explicit grids, and uses the
compiler resolver for explicit graphs and older libraries without the query.
A refused query never retries another policy. An omitted VV10 grid keeps its
own standard level for a nonstandard Treutler primary, but shares an explicit
legacy tensor grid by default.

## Invariants

Do not change SDK shared-by-omission, CLI Hessian, CUDA, or derivative
admission. The metadata query is not a capacity or property promise. It does
not retire the authenticated Python graph view or move resource planning.

Refs #2227.
