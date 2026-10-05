# Decision: publish the qualified DF-RCCSD(T) force owner

Status: implemented
Date: 2026-10-06

## Problem

The correlation-only DF-RCCSD(T) native CUDA owner already computes complete
analytic forces, including corrected Lambda, factor/source/metric response,
exact-reference orbital/Z/Pulay response and nuclear terms. The public
`df-rccsd(t)` method nevertheless advertised energy only and rejected force
requests before entering that owner. This left issue #158 looking like missing
gradient science after the scientific implementation and large-system
qualification had already landed.

## Decision

Expose `forces` in the existing `df-rccsd(t)` / `df-ccsd(t)` public method
manifest and forward the public `compute_forces` request to
`run_df_ccsdt_native`. Keep the same CUDA/FP64, conventional-RHF +
correlation-only-DF, explicit-auxiliary-basis method definition.

This is capability publication only. It does not add another force equation,
response solver, derivative consumer, precision mode, density-fitting
approximation, or CPU fallback.

## Rejected alternatives

- Reimplementing the native force chain in the public method owner was rejected
  because #1809 already composes the complete qualified owner.
- Keeping the public method energy-only until another large campaign was rejected
  because merged 230-AO complete-force and constrained-budget evidence already
  exercise the same native owner; another internal rerun would not validate a
  different scientific path.
- Enabling prepared batches in the same change was rejected because batch
  lifecycle is a separate public capability and is not required to publish
  single-system forces.

## Invariants

- The RHF reference remains conventional, exact and unscreened.
- Density fitting applies only to the correlation Hamiltonian.
- Public force requests use the same FP64 native owner and acceptance gates as
  retained internal qualification.
- Unsupported CPU, frozen-core, open-shell, alternate metric-threshold and
  separate DF-memory-budget domains remain fail-closed.
- The C2a source-level validation facade may remain energy-only; it is not the
  native Calculator provider.

## Evidence

- #1809: complete native DF-CCSD(T) force composition and small-system
  independent finite-difference/analytic validation.
- #1918 / #1927: post-merge complete water/ethane force integration,
  constrained-budget and retained independent force/FD checks.
- #1975: later 230-AO exact-response optimization on the same scientific force
  endpoint.
- Public registration tests bind the manifest to energy+forces and require the
  public owner to forward `compute_forces` to the native owner.
- An opt-in finite-Slurm-GPU public H2 smoke test compares the published force
  against public energy finite differences.

## Consequences

Users can request `properties=("energy", "forces")` from
`Calculator(method="df-rccsd(t)", device="cuda", auxiliary_basis=...)` without
changing the underlying scientific route. Broader performance work remains
separate from capability support.

## Revisit when

Prepared-batch DF-CCSD(T), frozen-core/open-shell variants, a DF-RHF reference,
or additional precision/metric policies receive their own qualification.

## References

- #158
- #1809
- #1871
- #1918
- #1927
- #1975
- `docs/developer/df_ccsdt.md`
- `docs/developer/df_ccsdt_gradient.md`

Agent: ChatGPT
Model: GPT-5.6 Sol
