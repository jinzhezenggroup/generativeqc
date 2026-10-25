# Decision: retain proven HF states and avoid dense DF source duplication

Status: implemented
Date: 2026-10-25

## Problem

The complete README campaign at `92674610bac506d76b762bc30594e72aae0a1baa`
exposed three separate costs. Fast direct HF at 96 atoms spent 42.55 s in a
second strict solve after its candidate had already completed Standard FP64
validation. The candidate recorded a retained final physical Fock, but
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

## Measured follow-up

The source-matched campaign compares baseline
`194f12d788d0073b4e158968d0eaa694283b0d34` with candidate
`1a971cfc508c666f09ee8490039e74b6843c54f0` on a Slurm-allocated RTX 5090.
The retained [publication](../../../../benchmarks/results/hf-cold-work-20261025/README.md)
includes complete cold endpoints, three frozen-density warm calls, all exposed
Fock work, independent numerical gates, source/binary identities and ablations.

Fast Direct HF at 48/96 atoms decreases from 17.719/60.892 s to 6.119/19.380 s,
with 31/32 total Fock builds falling to 10/11. A Python-only ablation on the
unchanged baseline native library reproduces the single-solve result. Explicit
observable-audit targets continue through the independent audit path.

Standard DF-HF at 48 atoms decreases from 14.735 s to 3.456 s, and its cold
Fock count falls from 23 to 11. Packing alone gives 4.486 s with the original
23 Focks, isolating the layout gain from retained-determinant finalization.
At 96 atoms, packing was already selected by capacity; the new final-state
proof reduces cold time from 28.496 s to 21.771 s and Focks from 21 to 12.
Its profiled finalization falls from 10.85 s to 4.19 s. Source preparation
remains about 6.49 s (4.54 s whitening projection and 1.94 s raw generation),
and SCF remains about 7.05 s. These are the next large-work targets.

All seven 48-atom DF-DFT methods also improve cold latency. Their Standard
before/after force changes are below 9e-12 Eh/Bohr. The historical DF-HF fitting
error against direct HF remains visible; it is distinct from the same-DF independent
reference and changed-geometry regression checks. Neither an unchanged failing
fitting gate nor a lower observed latency establishes universal qualification.

The retained qualification contains 521 passing Python tests, three passing
explicit CUDA audit tests, all four native test programs, 18 passing displaced
geometry/property calls, and zero device final-state memcheck errors. All 120
candidate samples also pass the same-model GPU4 comparison. Broad native testing
exposed an obsolete J-only fitted-panel count assertion, reproduced with the
unmodified baseline. Its replacement retains the independent force oracle and
bounded-panel coverage while tightening regenerated source bytes to at most one
complete pass; a fused charge provider may materialize none. This test-only
repair does not change the measured production source or library.

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
