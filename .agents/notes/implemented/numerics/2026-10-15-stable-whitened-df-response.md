# Decision: reverse whitened DF factors before quadratic metric seeds

Status: implemented
Date: 2026-10-15

## Problem

The practical-auxiliary extension of complete range-separated DF exposed a
force-component error of 4.41e-7 Eh/bohr on CPU and 4.83e-7 on CUDA, above the
unchanged 3e-7 gate. CPU warm forces varied by approximately 1e-6 although the
energies agreed. The negative evidence is retained in
`benchmarks/results/range-separated-df-20261015/`; this decision supersedes its
unresolved numerical limitation, not its measurements or build boundaries.

The raw AO/auxiliary Gram loses small spectral seeds when the LR metric retains
eigenvalues near its relative 1e-10 cutoff. Applying the pseudoinverse Frechet
map afterwards amplifies this cancellation by two inverse eigenvalue powers.
Tighter SCF convergence and different finite-difference steps do not repair
that representation of the reverse computation.

## Decision

Reverse `Z=A M^{-1/2}` instead: form whitened factors before quadratic products,
compute their energy adjoints, then contract `bar_A=bar_Z M^{-1/2,T}` and
`bar_root=A^T bar_Z`. Apply the existing inverse-square-root Frechet map with
the forward eigensystem and cutoff, including retained/discarded cross terms.
The forward Hamiltonian, cutoff, integral normalization and acceptance gates
are unchanged. CPU weighted and materialized derivatives share these weights.

CUDA retains one explicitly budgeted complete whitened tensor for truncated
metrics, with bounded raw and derivative-weight panels. Raw values are read
twice per auxiliary slice independently of the response tile; AO density
products repeat across response panels. This is not a constant-work claim for
arbitrarily small tiles. Insufficient capacity fails closed. Full-rank fitted,
occupied and resident response paths retain their prior qualified dispatch.
All arithmetic stays on the owning CUDA stream; no CPU/reference fallback is
introduced. Raw values cannot be reconstructed from a truncated fitted tensor:
discarded raw directions are needed for subspace motion.

## Rejected alternatives

- Loosening the force gate or changing the metric cutoff changes the scientific
  contract rather than fixing the derivative.
- Keeping the raw-Gram route as a low-memory fallback restores the failure.
- Retained-eigenvalue-only inverse identities lose discarded-subspace motion.
- Calling libcint/PySCF in production would hide an oracle dependency.

## Evidence and invariants

The initial CPU prototype reduced the same practical force-component error to
1.44e-9 Eh/bohr and warm replay deviations below 8e-10, with the same energy
error of approximately 1e-11 Eh. Final source-matched endpoint evidence belongs
in the superseding benchmark results, not in a rewritten historical receipt.
Executable host CUDA tests exercise both adjoints, signed Coulomb/exchange,
retained/discarded spectral motion, bounded tiles, immutable inputs and charged
workspace guards. Independent reconverged libcint endpoint oracles retain
2e-8 Eh energy and 3e-7 Eh/bohr component/directional force gates.

## Revisit when

A factor-space representation can retain full truncated-subspace response with
less than one AO-pair/auxiliary tensor, or measured small-tile density-product
amplification justifies a separately admitted reuse strategy. Preserve the
existing generated Frechet algebra and independent practical-auxiliary gates.
