# Decision: contract the DF metric derivative per auxiliary shell pair

Status: implemented
Date: 2026-11-18

## Problem

DF HF/hybrid forces contract `dM[P,Q]` with a full, nonsymmetric weight matrix.
The complete-panel route (2026-10-25 pair reuse) still evaluated the generic
primitive-product derivative once per unordered pair of auxiliary AOs. For the
README auxiliary on water-96 that is 89 million AO pairs and 115 ms per H100
warm call, against 7.7 ms for GPU4PySCF's `int2c2e_deriv`.

## Decision

- **Route:** one thread per ordered auxiliary shell pair, in 25 class kernels.
  The route needs a complete contiguous metric panel, every auxiliary shell at
  most g, harmonic AO polynomials (spherical, or Cartesian s/p) and the Coulomb
  operator.
- **Mathematics:** a harmonic degree-l polynomial obeys
  `S(r-A) exp(-a|r-A|^2) = (2a)^-l S(d/dA) exp(-a|r-A|^2)`. Each AO is therefore
  an exact combination of degree-l Hermite Gaussians, with the coefficients of
  its own normalized expansion.
  - The thread projects `W[P,Q]+W[Q,P]` onto those monomials.
  - It contracts the projection with the compiler-owned DF radial recurrence,
    raised by one order.
  - It differentiates only the first center; the ordered sum over both
    orientations is the complete gradient.
  - Same-atom pairs cancel exactly and are skipped.
- **Table:** the host builds the coefficient table from the same expansions as
  the derivative view and rejects any polynomial whose Laplacian is not zero.
- **Fallback:** partial, offset, strided, serial and distributed-sink panels,
  range separation, auxiliary h and Cartesian d and above keep the per-AO route.
- **Controls:** `GENERATIVEQC_DF_METRIC_DERIVATIVE_SHELLS=0` rolls back to the
  per-AO pair route; invalid values reject. The shell route is reached only
  while `GENERATIVEQC_DF_METRIC_DERIVATIVE_PAIRS` is unset or `1`, so its dense
  rollback is unchanged. The new control joins resource/checkpoint identity.
- **Coverage:** the standalone weighted API uses the same route when one
  uploaded tile holds the whole metric.

## Evidence

`benchmarks/results/h100-df-metric-shells-20261118/`. The same binary was run
under both controls, in order legacy, shell, shell, legacy.
- **Warm timings:** DF-HF96 0.85-0.86 → 0.73-0.75 s, B3LYP96 1.64 → 1.52-1.53 s
  and HF48 0.18 → 0.15 s.
- **Kernel time:** the metric kernels take 0.62 ms per call, down from 115 ms.
- **Force agreement:** same-phase forces agree within 1.8e-12 Eh/Bohr.
- **Independent oracles:** libcint `int2c2e_ip1` with arbitrary weights, s..g
  contracted fixtures in both representations, gives errors of at most 2.2e-12.
  PySCF DF-RHF auxiliary response on README water-12 gives errors of at most
  8.8e-12.
- **Tests and memcheck:** the full DF derivative GPU test file passes, and
  memcheck is clean.

## Revisit when

- A range-separated metric needs it. That requires an erf-attenuated radial
  oracle, since the radial lowering already accepts omega.
- Partial panels need it. Ownership of both orientations must then stay bounded.
- Auxiliary h appears in a production basis.
