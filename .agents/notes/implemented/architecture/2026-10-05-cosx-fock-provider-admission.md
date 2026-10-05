# Decision: retain response reservations when admitting COSX providers

Status: implemented
Date: 2026-10-05

## Problem

The enclosing COSX Fock composition passed only the pure exchange-buffer estimate
to its prepared value owner. Shared contraction candidates could not acquire any
additional capacity, even with an otherwise spacious complete endpoint budget.
Simply subtracting current Coulomb residency would instead let a retained
exchange provider consume the Coulomb response reservation before its buffers
were allocated.

## Decision

The existing Fock owner exposes its resolved additional response reservation and
resident-plus-response peak. The COSX composition still prepares J using the
budget after the base COSX value/derivative buffers are reserved. After J
preparation, charge actual J residency and the larger of the J and K transient
response bounds; both value owners remain resident, but synchronous J and K
responses do not overlap. Pass the remaining envelope to the shared exchange
owner without selecting a provider or changing a scientific request.

The complete peak includes the selected exchange provider's retained allowance.
Value and molecular response host binding reservations are recorded separately
from numerical host observations. Successful Fock builds refresh exchange work
diagnostics after all spin matrices are available.

## Rejected alternatives

- Pass the entire unused resident budget while ignoring future J response work.
- Require a method-local fixed vendor allowance, or force a provider in Fock code.
- Add both sequential response bounds and unnecessarily reject a legal endpoint.
- Treat the initial zero-call exchange snapshot as replay diagnostics.

## Invariants and evidence

Retain the same equations, generated production incumbent and exact derivative
contracts. A small envelope or unavailable optional provider retains generated
execution, including complete fixed-density derivatives. All real-device work
uses finite Slurm allocations and compiler caching.

The production enclosing Fock test compiles its value owner with test-only
qualification hooks; the production shared library has no new hook. It exercises
individual/full projection, accumulation and ESP masks, RHF/UHF full/tail work,
one shared provider reservation, constrained-budget/unavailable fallbacks and
independent RI-J/COSX value, energy and analytic molecular derivative oracles.
Source/binary manifests, compiler commands, ccache statistics and qualification
logs are retained in ignored `.artifacts/1884-cosx-fock-admission/`.

## Consequences and revisit

This admission repair permits provider competition through the real composition;
it does not promote a library profile or claim a speedup. Complete production
SCF/force crossover qualification and remaining derivative-region migration stay
under #1884 and #1886. Concurrent J/K response would require a different peak
contract before being enabled.

## References

#1884, #1886, #1970 and
[checked derivative sites](2026-10-05-cosx-checked-derivative-sites.md).
