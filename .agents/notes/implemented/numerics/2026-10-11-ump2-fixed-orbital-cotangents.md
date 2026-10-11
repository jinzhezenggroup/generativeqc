# Decision: shared-AD UMP2 fixed-orbital cotangent assembly

Status: implemented
Date: 2026-10-11

## Problem

UMP2 energy already has a compiler-owned differentiable spin-labelled TensorIR
equation, but RHF-only adjoint assembly cannot preserve unrestricted spin counts
or the same-spin one-quarter coefficient. Complete UMP2 forces need independently
qualified unrelaxed MO weights before orbital response and derivative consumers.

## Decision

Reuse the existing shared VJP and keep three-channel MO assembly in a separate
internal module. Tile results retain distinct direct/exchange and four energy
feeds; canonical assembly maps exchange weights back into the original physical
tensor and accumulates both spin energy vectors. Identity strings are explicit
caller attestations, matching the existing canonical RHF weight API boundary.
This interface owns neither a UHF solver nor AO transformation/response policy.

Borrow already-FP64 inputs and publish immutable non-aliasing weights. Numeric
admission charges dense outputs and publication copies/scratch as well as full/tail tile
workspaces; finite scanning is bounded. Validate all active-channel denominators
before any AD call. No result escapes on a later failure.

## Rejected alternatives

- A second hand-written production derivative formula would duplicate the
  compiler's scientific owner and drift from its factors/provenance.
- A thin wrapper around one full-block VJP would omit disjoint rectangular
  exchange assembly and give no pre-execution global output budget contract.
- Integrating UHF relaxation or native forces now would obscure this independently
  reviewable prerequisite and require additional scientific acceptance gates.

## Invariants and evidence

`tests/python/test_ump2_adjoint.py` independently enumerates all ordered
spin-orbital excitations with spin Kronecker deltas, including beta-alpha pair
interchange and both virtual spin exchanges. Seed 1823 defines asymmetric
open-shell shapes; independent directional seeds and per-feed perturbations
detect sign/factor errors. The restricted limit sums all three integral bars and
both energy bars before comparing to RHF perturbations. Li/open-shell and H2/
broken-symmetry reference tests require pinned PySCF and cross-check their oracle
energies against PySCF UMP2; NumPy-only runs disclose these skips.

Budget boundary tests forbid both output allocation and AD on rejection. Late
failure injection checks that previously accumulated tiles cannot escape and
that inputs are preserved. Four primal numeric capacities are conservative for
the fixed add/broadcast/divide/multiply/reduce VJP inventory, not a generic
process-memory promise.

## Revisit when

A streamed provider/consumer can avoid dense canonical outputs, or qualified UHF
response and derivative consumers are ready. Preserve independent MO derivative
evidence and the shared equation authority while introducing either consumer.

## References

Refs #1823; current contract: `docs/developer/ump2_adjoints.md`.
