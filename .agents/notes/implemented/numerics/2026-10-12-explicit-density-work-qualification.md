# Decision: qualify density-only LDA with an explicit work-input contract

Status: implemented
Date: 2026-10-12

## Problem

The unclipped bulk v2 candidate and the automatic point executor have different
input contracts. In the pinned CHACHIYO polarized zero/full-spin rows, raw
derivative roots differ from PySCF 2.14.0 / Libxc 7.0.0 by approximately
2.18–2.31e-6. Evaluating the same roots at the already-declared 1e-15 per-spin
work floor recovers the complete reference potentials, including the minority
coefficient. This diagnosis does not establish physical particle-birth behavior.

## Decision

Keep interior, v1, and unclipped v2 mathematics unchanged. Add the distinct
`libxc-7.0/work-density-only-v1` domain (native ABI version 5). Construct interior
energy/derivative roots first, then substitute the pinned safe density inputs.
Do not differentiate the clipping map. Weight energy with original total
density, not the work spin sum. Screen total density before evaluating any roots.
Divide energy by work density before multiplying by original density and apply
this projection after interior algebraic optimization. Reflattening its quotient
into a multiply-first n-ary product spuriously squares densities and overflows
around 1e160 although the original unpolarized energy is finite. Protect this
gate in every supported optimization mode.

Select this qualification recipe structurally for density-only LDA; preserve
the existing unclipped recipe for other ingredients. Production-domain,
compiled-CPU, molecular-SCF and catalog campaign producers share the selector.
The molecular producer blocks this domain before execution until native
restricted normalization is separately qualified. Independent pinned checks
confirm that polarized folding at total rho=1.1e-15 differs from the unpolarized
potential by approximately 2e-6 for representative LDAs. Routine molecular
geometry passes observed during diagnosis do not certify that tail. Retain the
full RKS/UKS requirement rather than dropping RKS rows or substituting spin-1
reference outputs for a restricted endpoint.
The new expression binds the worker policy and projection owner; prepared
programs capture that owner identity once. Work evidence readers rebuild the
current program and reject stale source/expressions rather than relabeling old
raw evidence. Historical raw receipts stay as diagnostic artifacts, not new
source-qualified evidence.

The CUDA tile emitter must implement density screening for derivative outputs,
not just energy-only vacuum. Otherwise a mixed batch would execute floored
potentials at screened points even though array and native point paths return
zero. Domains without a density threshold keep their original generated guard.

## Rejected alternatives

- Change v2 silently: invalidates its numerical meaning without a new identity.
- Differentiate a clipped energy: yields the clipping derivative, not Libxc's
  interior potential evaluated at work coordinates.
- Hide the minority channel or loosen the acceptance gate: loses a scientific
  output and does not fix the mismatched input contract.
- Extend this projection to sigma/tau: density work alone does not reproduce
  those workers' clamps, curvature policy, or physical-input predicates.
- Modify the existing automatic work owner: unnecessarily changes automatic
  emitted science for a qualification-consumer correction. The conservative
  compiler inventory still invalidates all source-bound capability/expression
  identities; do not suppress that invalidation to reuse old evidence.

## Invariants and acceptance gates

- Preserve old domain mathematics and distinct contracts; no name-specific branch.
- Require complete E/vxc, including empty-spin minority coefficients.
- Keep nonnegative input validation, original-density energy weighting and
  pre-evaluation screening; do not mutate input arrays.
- Rebuild work receipts against current source/domain/worker identities.
- Use pinned independent full-output controls and complete native molecular
  receipts, with observed positive J and zero K for pure semilocal campaigns.
- Real CUDA tests run under finite Slurm allocation with scheduler visibility.

Executable gates are `test_bulk_density_work_contract.py`, the opt-in
`test_bulk_density_work_native.py`, and the existing production-domain,
compiled-CPU and molecular-SCF qualifiers. The native point contract is polarized;
standalone unpolarized work coordinates remain a distinct normalization. In
particular, folding RKS total densities in `[floor, 2*floor)` into polarized
coordinates can change the worker evaluation. Do not infer RKS tail equivalence,
forces, physical birth, public CUDA admission, or full-catalog closure from a
passing polarized work test.

The one-off migration comparison covers 154 emitted science sources: all eleven
representatives, both backends, automatic points and both spin layouts in each
old bulk domain. Their emitted bytes are unchanged. All 154 source-bound
expression identities conservatively invalidate, as required by the existing
compiler inventory. This is retained migration evidence, not a permanent test
that pins production source hashes.

## Revisit when

Independent physical-boundary and complete-endpoint gates justify extending the
worker projection to other ingredients or changing restricted point normalization.
Such changes need their own explicit contract and evidence, not a hidden floor
adjustment or an oracle wrapper chosen to make an existing test pass.

## References

- #1118, #1119, #1121 and #1124.
- `docs/developer/libxc_bulk_capabilities.md` for the current qualification contract.
- The preceding raw/work diagnosis is retained in the owned qualification
  artifacts under `.cache/1118-runtime/molecular-jk-evidence-phase-pr-20261012/`.
