# Decision: borrowed structured SCF region bindings with closed default gates

Status: implemented
Date: 2026-10-11

## Problem

Moving only the vendor submissions does not give density, orthogonalizer and
basis stages a common semantic owner. Replacing their kernels immediately would
also confuse primitive qualification with a complete scientific endpoint gate.
The standalone rank-k owner cannot simply be instantiated in each SCF stage:
it owns provider/storage resources already owned by the SCF arena and context.

## Decision

Reuse the existing TensorIR rank-k recognizer for ordinary and signed density,
and the existing four basis equations. Inverse-root reconstruction retains its
explicit power expression. Emit bounded portfolios with negative offers and
freeze them in an immutable borrower of the existing stream/cuBLAS context.
Native preparation resolves actual dimensions and byte charges. It does not
invent timings. The final Fock request includes its hcore publication epilogue.

RHF/UHF and KS consume shared density bindings; conventional HF also consumes
orthogonalizer and dense basis-stage bindings. Provider glue is tensor-owned.
KS shell-span transforms remain in their current sparse owner. Production
library eligibility is false until the full scientific and endpoint gates are
met. Diagnostic test-hook bindings admit library ablations independently.

Reuse dead solver/back-transform scratch for column scaling and the charged
masked-publication span. Signed KS reconstruction additionally borrows the dead
effective-Fock plane only after the final-frame solve/proof. Preparation has a
bounded 32 KiB host charge; provider storage remains under its existing owner.
The optional basis workspace is an arena resource fact, not a planning-layer
dependency on a CUDA binding. It allocates no bytes with the closed default gate.

## Rejected alternatives

- Multiply a sanitized weighted panel by the original coefficients for signed
  density: unoccupied coefficients may be NaN, and `0 * NaN` contaminates GEMM.
  Both panels must exclude unoccupied/inactive columns before submission.
- Use square-root scaling for signed weights: this changes the real-valued
  domain of energy-weighted density. Signed reconstruction uses GEMM instead.
- Advertise a ready rank-2k recipe merely because cuBLAS has SYR2K: none of these
  seven regions is a symmetric pair. Keep negative evidence instead.
- Pack sparse shell spans to dense to force BLAS: this discards bounded work.
- Promote defaults from isolated arithmetic or timing: final SCF and force
  gates and complete cold/warm endpoints are independently mandatory.

## Invariants

- No selection, allocation, discovery, CPU oracle or new provider owner during
  production iteration/capture.
- Masks are device-live; rejected/inactive outputs retain their bytes.
- Eigenvalue validation rejects nonfinite or `<= 1e-10` values per item and
  remains independent of the reconstruction provider.
- Positive Gram publishes a mirrored lower triangle. Signed weights and
  inverse-root use GEMM; their published symmetry follows the same convention.
- Scratch spans are bounded and disjoint before any library work is enqueued.
- AOT identities describe templates; borrowed runtime dimensions/context and
  stream-live storage remain mandatory executable facts, not cache keys.

## Evidence and remaining gates

Host tests exercise canonical equations with an independent NumPy oracle,
rank-k recognition, explicit negative offers, owner lifetimes and resource
charges. The Slurm-only native driver uses a diagnostic long-double CPU oracle
and replays the same Graph after changing pointed-to data and masks. Raw source,
dependency, compiler/cache and device receipts are retained in ignored
`.artifacts/structured-scf-routes-20261011/`.

This does not close #1877, #1878 or #1883 by primitive evidence alone. Default
promotion still needs source-matched RHF/UHF/RKS/UKS energy/Fock/density/force
gates, through-d/f and moved-geometry coverage, complete cold/warm endpoint
costs, and production span-density diagnostics/admission for KS basis stages.

## Revisit when

Complete scientific endpoint evidence admits explicit per-region crossover and
resource facts. Preserve generated/sparse fallbacks and publish the evidence
before changing `production_library_qualified`.

## References

- `docs/developer/lowering_providers.md`
- `python/generativeqc_compiler/tensor/structured_scf.py`
- `src/scf/cuda/structured_matrix.cu`
- `tools/qualify_structured_scf_routes.sh`
- Parent decision: `2026-10-11-prepared-matrix-product-routes.md`
