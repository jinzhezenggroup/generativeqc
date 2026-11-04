# Decision: one native owner for CBS extrapolation coefficients

Status: implemented
Date: 2026-11-04

## Problem

Native HF/MP2/CCSD(T) components and native correlated stopping policies did not
retire Python's scientific CBS extrapolation arithmetic. Reimplementing that
arithmetic separately for SDK/CLI consumers would create another scientific
policy owner before the complete CBS execution endpoint migrates.

## Decision

`src/core/cbs_extrapolation.hpp` owns the two low/high weight pairs and the six
ordered focal-point coefficients. A copied, versioned C query exposes that pure
host plan without creating a context, looking up bases, compiling or executing
electronic work. Python requires the normally selected native library and uses
the same selection for the existing convergence-policy query. Missing support,
refusal and invalid receipts fail rather than reconstructing Python weights.

Preserve the incumbent stable `expm1` formulation. Finite positive exponents may
produce a negative-infinite decay intermediate: its limiting pair `(0,1)` is
valid. Singular denominators and nonfinite returned weights reject atomically.
Read the entire descriptor before publishing so borrowed buffers may overlap.

## Rejected alternatives

- Retaining an optional Python formula would leave the production owner
  ambiguous and conceal an incompatible selected library.
- Moving grouping, budget division or reductions in this cutover would introduce
  separate floating-point bracket and composition changes. Preserve existing
  method/basis grouping, `math.fsum` reductions and endpoint publication instead.
- Calling a coefficient query a native CBS endpoint would hide the remaining
  Python controller and unproved SDK/CLI and performance requirements.

## Invariants and evidence

- Real public C ABI tests use independent 80-digit decimal inverse-power and
  exponential models, ordinary/close cardinals, tiny and huge finite exponents,
  atomic malformed/null/header refusals and overlapping buffers.
- Executable Python tests inject missing/refused owners before electronic work
  and a coherent different native plan that must control component coefficients,
  full energy/force composition and convergence-budget amplification.
- Physical CBS tests retain pinned independent PySCF component/composite energy
  and full-force acceptance gates. No new backend, precision or derivative domain
  follows from this metadata-only cutover.

## Consequences and revisit conditions

The C ABI is reusable but Python still owns basis identity/family validation,
grouping, coefficient-L1 budget allocation, sequential component execution and
observable composition/publication. Keep #2227 R1–R6 incomplete. Revisit this
boundary when the existing shared prepared endpoint can own complete CBS work
and its transactional lifetimes with independent scientific and whole-endpoint
work/resource/performance gates; do not infer a speedup from constant-size planning.

## References

- `docs/user/ccsdt_cbs.md`
- `tests/python/test_cbs_extrapolation.py`
- `tests/python/test_ccsdt_cbs.py`
- `tests/python/test_ccsdt_cbs_native.py`
- #2227 / #2972
