# Decision: retain proven HF states and avoid dense DF source duplication

Status: implemented
Date: 2026-10-25

## Problem

The complete README campaign at `92674610bac506d76b762bc30594e72aae0a1baa`
exposed three separate costs. Fast direct HF at 96 atoms spent 42.55 s in a
second strict solve after its 19.88 s candidate had already completed Standard
FP64 validation. The candidate recorded a retained final physical Fock, but
Python fallback admission incorrectly required a warm initial seed.

At 48 atoms, DF-HF performed 10 SCF Fock builds followed by 13 physical final
builds. Its retained DIIS orbital columns failed column-wise canonicality even
though the occupied determinant could satisfy the physical force contract.
Repeated projection changed the determinant generation and lost the original
occupied-response lease. A separate diagnostic run attributed about 4.4 s to
dense source preparation: 3.3 s of raw values and 1.1 s of metric transforms.

## Decision

- Admit a completed strict Standard HF fallback using the native retained-Fock
  proof on either cold or warm calls. Keep strict arithmetic, Standard-or-tighter
  controls, default-or-looser budgets and explicit-target exclusion. This grants
  no per-call observable error certificate and invents no mixed work.
- Let the original CUDA DF-RHF force determinant pass through a fresh physical
  projector check without requiring its lagged orbital columns to be canonical.
  Preserve maximum/RMS projector defects and physical commutator, density
  reconstruction, metric orthogonality, trace and idempotency gates. Construct
  W from the same current `D F[D] D / 2`. Rejected probes feed bounded canonical
  correction; energy-only, reference export, forced rebuilds and correction
  steps retain their canonical checks. This is distinct from UHF stationary
  reuse, which can omit the projector probe.
- For already authorized singleton physical DF sources, consider the existing
  fitted-only packed owner once dense source storage reaches 1 GiB. Require the
  same value allowance and the full requested occupied projection. The existing
  streamed-to-resident capacity fallback, explicit selectors, small dense
  owners, unqualified general densities, UHF and multi-item boundaries remain.

## Rejected alternatives

Deleting the explicit observable audit would change the user's accuracy
contract. Omitting RHF's physical projector would lose the small-gap protection.
Canonicalizing an already proven occupied subspace changes the determinant and
can discard a useful response lease. Selecting packing solely for fewer bytes
could lose complete U or regress small warm endpoints; the resident crossover
therefore has both a work floor and an occupied-capacity gate.

## Validation contract

Use complete fresh-process cold energy plus analytic-force endpoints and frozen
engine-local warm densities, with independent oracles and actual Fock work.
Keep profiling separate from timing. Validate cold/warm/moved geometry,
bounded-resource fallback, strict controls, physical-reference export, stale
identities and small-gap projector rejection. Default README quadrature failures
remain visible and must be distinguished from regressions introduced here.

## Revisit when

Reassess the 1-GiB work floor on other devices or with changed source/response
costs. Broader batching, spin or general-density promotion needs its own complete
consumer evidence. Do not infer a universal speed advantage from memory savings.

## References

- `src/scf/solver/final_state.cpp`
- `src/scf/density_fitting.cpp`
- `python/generativeqc/precision_policy.py`
- `docs/developer/df_final_state.md`
- `docs/developer/df_packed_storage.md`
