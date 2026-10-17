# Decision: use orthonormal projector norms for UHF root equivalence

Status: implemented
Date: 2026-10-17

## Problem

Issue #1826 requires meaningful root equivalence and spin diagnostics. Recovery
used `||delta_D S||_F`. Although `delta_D S` transforms by similarity under an
AO basis change, its Euclidean Frobenius norm is not similarity invariant.
Rescaling two AOs by 1000 and 0.001 changes an otherwise identical root test:
a rotation of 1e-6 radians has physical projector distance 1.4142e-6, below the
1e-5 root tolerance, yet the mixed-AO norm incorrectly separates the roots.
The reported spin-density norm had the same coordinate dependence.

## Decision

Factor the positive-definite overlap once as `S = L L.T` after existing budget
and stability admission. Compare each spin with `||L.T delta_D L||_F`, and use
the same norm for the alpha-minus-beta density diagnostic. Any two such
orthonormal AO frames are related by an orthogonal transformation, preserving
the Frobenius norm. Nonfinite norms and failed factorization fail closed.
The factor and bounded matrix products fit within the existing conservative
36*nbf^2 recovery workspace reserve; no per-root overlap factorization is added.

## Rejected alternatives

- Keep the mixed-AO norm and enlarge the tolerance: basis conditioning has no
  fixed bound and would change the physical root-equivalence threshold.
- Compute squared distances as differences of occupied-overlap traces: this
  introduces avoidable cancellation precisely for nearly identical roots.
- Compare coefficient entries directly: orbital signs and occupied-space gauge
  rotations do not identify different electronic roots.

## Invariants and evidence

Preserve the existing per-spin root tolerance, equal-population-only global
spin-flip policy, source/Hamiltonian/certificate and stationary-energy gates.
This changes diagnostic geometry, not UHF equations or SCF convergence criteria.
Independent physical-frame projector tests cover scaled and sheared AO bases,
near/equal/distinct roots and spin inversion. A pinned PySCF 2.14.0 molecular
occupied-subspace/spin-square oracle and the existing Hubbard recovery and UHF
stability suites provide scientific checks. No UKS, geometry-tracking, public
force-selection or CUDA qualification is implied.

## Revisit when

A future non-idempotent/finite-temperature or cross-geometry root contract is
introduced: it needs its own scientific metric, overlap transport and acceptance.

Recovery integration uses AO transforms that pass the unchanged absolute
Hamiltonian and fresh-response residual gates. Manual probes with extreme
rescalings/shears failed these existing gates: mathematically invariant root
norms cannot authorize publication when the response itself loses accuracy.
Permanent negative gates inject explicit Hamiltonian/action defects rather
than depending on a BLAS implementation's rounding error. Primitive metric and
molecular projector oracles still exercise extreme coordinate changes directly.
