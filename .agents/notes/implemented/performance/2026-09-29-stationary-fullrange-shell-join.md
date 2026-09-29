# Decision: bind ordinary stationary CUDA J/K to prepared Direct shell derivatives

Status: implemented in source; NVIDIA numerical/performance qualification pending
Date: 2026-09-29

## Problem

The shared prepared Direct facade already contracts full-range J/K derivatives,
but ordinary semilocal/global-hybrid stationary forces still enumerate one or two
rectangular public-AO rank-four domains. The RSH migration does not fix that join.

## Decision

A token-checked snapshot ABI borrows the current KS device density and prepared
Direct derivative owner. Scientific J/K coefficients and spin conventions come
from the prepared Fock model, not a functional-name selector or Python formula.
The two compact source-major gradients seed a freshly reset stationary source
arena. Migrated sources are skipped before construction of any AO rank-four domain.
The existing compiler-generated source reduction still computes the final gradient.
No integral recurrence or second Direct owner is introduced.

The native source seed is synchronous and records its compact H2D and drain. The
new seed operation is not advertised as a zero-transfer handoff. It cannot replace
sources after tasks/geometry have launched, cannot execute twice per reset, and
must drain even partially submitted uploads before caller-owned host arrays die.

## Bounds and evidence

The native bridge bounds temporary staging to three two-source coordinate vectors.
Including the Python output gives at most 24*Natom FP64 values during the provider
call. The later seed stage holds 6*Natom plus 3*Nsource*Natom FP64 values. Both
finish before publication, reusing the existing conservative 120*Natom-double
transient host allowance; retained shell device state is already charged to SCF.

Descriptor work excludes migrated ERI sources; old AO quartet numbers remain
logical capacity, while public_ao_quartets_submitted records actual driver work.
Native screened/compacted/executed shell counts remain unavailable, not inferred
from capacity, timings, or the method name. CUDA endpoint speedup is unmeasured.

Device-free tests execute the real adapter with native stubs and the actual generic
driver with JIT/AOT mocks. The retained independent molecular CUDA gates remain
required for numerical qualification, including changed geometry and both spins.

## Fallback and rejected alternatives

Missing native/seed ABI or native NOT_IMPLEMENTED preserves bounded AO execution.
Stale tokens, invalid model/density, CUDA, allocation and numerical failures do not
fall back. ECP and RSH/nonlocal composition keep their existing independent routes.

A host-side ad hoc final sum and a new mandatory JIT reduction were rejected:
covered AOT execution must keep working without an installed runtime compiler,
and source ordering/ownership must stay with the stationary compiler.

## Revisit when

A shared native same-stream source-buffer lease can remove the compact source
D2H/H2D round trip while preserving ownership, package identity and failure drains.

Refs #1477, #1478, #1423. Builds on merged #1558/#1562.

Agent: ChatGPT
Model: GPT-6 Astra Pro
