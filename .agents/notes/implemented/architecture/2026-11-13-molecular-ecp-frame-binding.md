# Decision: carry the complete ECP Hamiltonian into every native CLI frame

Status: implemented
Date: 2026-11-13

## Problem

The native CLI saddle evaluator created each trial geometry with the
all-electron `System` constructor. Its public ECP refusal currently prevents
execution, but removing that refusal alone would silently change the electron
count, ionic charges, and residual potential. The Hessian and ordinary energy
entry points also constructed their local-basis systems independently.

## Decision

Use `cli::make_molecular_system` for these three local-basis consumers. It
validates the optional core-count layout and selects native ECP construction
when either residual potential terms or nonzero core counts are present. The
native constructor remains responsible for validating the physical records and
copying arrays into the owned system. Context lifetime remains unchanged.

Nonzero core counts without required residual potentials are invalid, not an
all-electron fallback and not a newly supported core-only Hamiltonian. Empty
potentials with all-zero core counts retain ordinary all-electron construction.
The separate method-basis path keeps its existing ownership.

## Invariants

- A fresh frame binds its own geometry and the same complete basis/ECP records.
- Later mutation of the caller's core/potential arrays cannot modify an existing
  native system.
- Native potential validation is not weakened to accommodate a helper test.
- Python and native CLI molecular ECP saddle admission remain closed until the
  real-element molecular consumer is separately qualified. This fix supplies
  correct construction, not a positive TS, scale, or chemistry receipt.

## Evidence

`generativeqc_cli_molecular_system_tests` executes a minimal two-electron Na/H
ECP RHF energy and all forces, comparing the helper with explicit native ECP
construction. It checks changed geometry, owned-array lifetime, all-electron
parity, malformed core layouts, and fail-closed core-only records. The minimal
potential is an ownership contract fixture, not a real-element oracle.

The initial test incorrectly expected core-only admission. Its failed exit and
debugger output remain under
`build/qualification/host-ecp-saddle-frame-20261113`; the corrected test checks
native refusal. No failed receipt is relabeled. The corrected test and five
additional native API/source-binding/Hessian/CLI contract groups pass against
the new CPU build. Older installed Ni/Cs numerical results retain their older
source identities and are not restamped onto the changed tree.

## Revisit when

Real I/Cs ECP molecular saddle consumers, including geometry updates, state
matching, and final raw Hessian/mode checks, pass their independent acceptance
gates. Only then consider changing the separate public admission policy.
