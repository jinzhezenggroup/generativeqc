# Proposal: compose conservative Direct RKS with merged CUDA defaults

Status: rejected; superseded by the user-approved portable default
Date: 2026-11-04

## Superseded design hold

The user explicitly approved portable 1024-point defaults and making #2948
Ready for review on 2026-11-04. The [implemented decision](../implemented/performance/2026-11-04-portable-direct-ks-tiles.md)
retains #2949's ordinary CUDA semilocal 1024/1024 defaults, including Direct,
with resource promises and bounded allocation fallbacks. The earlier mandatory
profile/256-point Direct policy below is retained only as rejected rationale;
its Draft hold is closed, not a condition for future integration work.

## Problem

PR #2949 merged as `9cfb77727661cc64e47938d9d0ba65a157427439`, giving ordinary
semilocal CUDA owners 1024-point SCF and force defaults. The separate Draft
#2948 removes hardware-generation-based Direct promotion. Choosing either side
of their conflict wholesale would erase the Direct hold or silently revert
already merged fitted/unrestricted defaults.

## Proposed composition

Keep the merged semilocal, nonlocal, explicit-request and bounded-retry behavior.
Pass Direct RKS identity separately to the pure SCF policy. Non-fitted RKS stays
at 256 automatically, without GPU names or SM80 capability assumptions; an
explicit tile or independently qualified exact-workload profile still takes
precedence. Its automatic semilocal force tile inherits the resolved 256-point
SCF tile, while the fitted nonlocal force policy remains independent.

## Invariants and validation

Host-compiled policy contracts cover Direct across unknown/older/SM80/Hopper/
consumer-GPU capability tuples, explicit overrides, general non-Direct 1024,
fitted nonlocal 512 and constrained 256. They execute no real GPU work and are
not performance qualification. Numeric admission, allocation retry and all
independent scientific gates remain unchanged.

## Historical hold (superseded)

The PR remains Draft. The previously documented A100 default-regression/design
question is unresolved; no fresh GPU endpoint measurement is requested or
performed in this composition. Historical measurements remain frozen and are
not relabelled as evidence for this integrated source.
