# Decision: revalidate Fast's stationary warm seeds directly in FP64

Status: implemented
Date: 2026-10-20

## Problem

Default-grid README measurements at source `02337e0cf` found Direct PBE/PBE0
warm Fast running two mixed and two strict Focks, while Standard took one
strict Fock. The approximate restart perturbed an already converged same-owner
seed. The prior strict warm selection covered experimental mixed XC but omitted
application budget-Fock. Merely bypassing native mixed work would also fail the
Python controller's positive-FP32-work admission and trigger another endpoint.

## Decision and invariants

Extend the existing same-owner warm FP64 selection to budget-Fock. Keep energy,
density and physical residual gates, finite iteration limits, final-state
validation and cold/external/moved seed handling. This changes arithmetic
selection, not stopping tolerances or requested scientific outputs.

The Python policy separately recognizes strict prepared warm results only with
complete operator inventory, no executed mixed work, positive strict Fock work
and a final physical audit bound to the returned state identity. Required
operators follow the actual model, including exchange and nonlocal correlation.
No synthetic mixed work or per-call observable error is reported. Missing or
inconsistent evidence retains the full audit fallback. An explicit accuracy
target always retains its independent endpoint audit.

## Evidence and remaining qualification

The policy/batch host suites pass 178 tests, including missing inventory,
approximate arithmetic, missing exchange, stale state, cold/fallback and explicit
target rejection cases. The new default-grid real-device test checks independent
CPU energy/forces for frozen and moved 12-atom PBE/PBE0/r2SCAN/M06-2X endpoints.
Real-device execution and complete endpoint performance are pending at this
checkpoint; no universal Fast speedup or README production ranking follows.

Revisit source reuse and force scheduling if eliminating the mixed restart still
leaves warm Fast slower. Keep those measured causes separate from the stopping
policy and from independent audit requirements.
