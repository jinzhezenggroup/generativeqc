# Decision: measure CPU KS Coulomb/exchange applications at the provider

Status: implemented
Date: 2026-10-12

## Problem

Automatic semilocal CPU endpoints have numerical lifecycle evidence, but an
uninstrumented precision inventory cannot prove absence of unused exchange.
Neither an absent exchange coefficient nor the iteration count is execution
evidence. Full precision instrumentation is a different, broader contract.

## Decision

Carry a narrow execution observation with raw J/K matrices. Exact GEMV and
specialized contractions, and the CPU density-fitting application calls, record
only operations that actually execute. Independent provider composition and
the KS physical-evaluation owner aggregate observations. The additive native
v1 query exposes this separate scope without changing existing descriptor
sizes, result strides or precision completeness flags.

J counts applications to the total density. K counts each executed spin,
including zero-density and zero-coefficient contractions. This is a value-side
target census, not recurrence/FLOP counts, derivative work, initial integral
preparation or an endpoint speedup claim. Preliminary-method work is excluded,
while discarded target attempts and final physical-Fock closure are included.

## Invariants and rejected alternatives

- Do not derive any work count from iterations, coefficients or method names.
- Default observations are unknown, not certified empty. A missing provider,
  lost exceptional attempt or overflow permanently revokes completeness.
- A cold retry cannot certify missing work from a discarded attempt.
- A complete narrow J/K census never certifies the full precision inventory.
- CUDA/COSX and older native libraries remain explicitly unavailable here;
  their independent execution evidence is not synthesized or overwritten.
- Numerical contractions, floors, tolerances, occupations and SCF budgets are
  unchanged; no new named-functional science or reference runtime is added.

## Evidence

Native provider tests check exact/DF, independent/shared providers, both spins,
every J/K selection, zero coefficients, overflow and unavailable observations
against the existing independent four-index numerical oracle. CPU endpoint
tests compare observed J to actual physical-Fock builds and observe zero K for
all 11 automatic registrations, both spins, cold/warm/changed geometries and
exhausted budgets. Preliminary retry tests are explicitly control-flow doubles,
not numerical qualification.

## Consequences and revisit conditions

Consumers can distinguish an actual CPU J-only observation from unsupported
full-inventory data. Extend this census only when a new provider records its
own executed operations; qualify derivative/recurrence/precision scopes
separately rather than broadening this record by inference.

## References

- #1118 and #1121
- `docs/developer/ks_diagnostics.md`
- `tests/native/test_fock_provider.cpp`
- `tests/python/test_libxc_automatic_cpu_ks.py`
