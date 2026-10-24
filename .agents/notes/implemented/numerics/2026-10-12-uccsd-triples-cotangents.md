# Decision: differentiate unrestricted triples on the shared bounded tile grid

Status: implemented
Date: 2026-10-12

## Problem

Complete canonical UCCSD(T) forces need triples amplitude, Hamiltonian and
denominator bars. Unrestricted triples has four independently owned spin
sectors; a full T3 embedding is not an acceptable production response buffer.
An energy-only tile budget does not admit derivative work or publication
storage. Standard UCCSD Lambda stationarity is not the corrected method's
stationarity equation.

## Decision

Apply generic TensorIR reverse AD to the existing physical scalar tile graph,
including every explicit orbital-energy operand. Reverse the compiler-owned
bindings with compensated host scatter into full canonical source bars. The
physical excitation gradient uses U*, including all signed same-spin orbit
members. One shared energy/response owner controls fresh CCSD qualification,
denominator checks, full/edge shape templates, backend budgets, grid coverage,
deterministic fallback and transactional publication.

Keep this response record distinct from standard Lambda or nuclear results.
Its explicit eps bars are not silently discarded or silently attached to
standard Fock weights. The explicit `hamiltonian_cotangents()` coordinate
pullback adds them to their own spin/occupied/virtual Fock diagonals. Corrected
Lambda and the full physical pullback remain required consumers. The standard
physical response owner must
continue rejecting multipliers which are not standard stationary UCCSD Lambda.

## Rejected alternatives

- Full molecular T3 or full spin-orbital production tensors: defeats the
  existing bounded tile contract and duplicates canonical source storage.
- Handwritten unrestricted derivative inventory: duplicates scientific algebra
  and can omit a spin sign, a repeated binding or the physical denominator.
- Reuse energy-only admission: derivative work, source compensation and
  immutable publication all need explicit admission before native execution.
- Treat corrected multipliers as standard Lambda: bypasses the very
  stationarity gate needed to define complete analytic UCCSD(T) response.

## Invariants

- aaa/aab/abb/bbb all differentiate the same qualified (T) energy definition.
- Spin-owner axes and mixed-spin antisymmetry signs remain in the existing
  graph/bindings. No fixed-owner mixed-spin pair exchange or restricted factor
  is inferred by the scatter.
- CUDA uses its matching native equation executor, not CPU equations; host
  binding, sums and coordinate maps remain explicit roles.
- Numeric admission is logical buffer accounting, not an RSS/allocator claim.
  Smaller tiles still execute the full grid and obey its scalar-work bound.
- Late failures publish neither prefix energies nor any partial cotangent.
- Fixed-amplitude derivative closure does not qualify complete nuclear forces.

## Evidence

`tests/python/test_uccsd_triples_response.py` compares all four channel energies
and amplitude/Fock/integral/denominator directional derivatives to pinned
PySCF's independent triangular GCCSD(T) implementation. The oracle enumerates
distinct virtual triples, not production's ordered six-axis grid. Complete
native endpoints cover asymmetric H6, NH2 and broken-symmetry H4, full/edge
scatter parity, signed amplitude pullback, immutable detached records,
deterministic memory fallback, pre-execution admission, empty domains and
injected late native/nonfinite/malformed-output failures. A finite Slurm tier
requires real CUDA and forbids native CPU equation execution.

## Consequences

Full conventional source bars remain O(N^4), while T3/AD moments are tile
bounded. This is not a performance promotion. The corrected response chain
must budget these retained bars and qualify fresh corrected stationarity and
reconverged total-energy force closure before claiming issue #1825 acceptance.

## Revisit When

An independently qualified streamed physical pullback can avoid retaining full
canonical bars without changing the mixed-spin, denominator or transactional
contracts; or a separately bounded DF implementation is scientifically ready.

## References

- #1825: complete canonical UCCSD(T) analytic forces.
- `docs/developer/uccsd_triples.md`.
- `docs/developer/uccsd_t_owner.md`.
- `docs/developer/uccsd_physical_response.md`.
