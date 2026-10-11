# Decision: sum UCCSD spin sectors into one differentiable TensorIR graph

Status: implemented
Date: 2026-10-12

## Problem

Issue #1821 requires five unrestricted amplitude blocks with independent orbital
populations. The existing RCCSD inventory and pair packing do not encode the
independent occupied/virtual antisymmetries or mixed-spin multiplicities.
Separate CPU/CUDA equation implementations would also duplicate the future
Lambda/Hamiltonian derivatives required by #1824 and #1825.

## Decision

Build canonical spin-orbital energy and physical residual equations with exact
rational coefficients, then eliminate forbidden spin assignments at graph
construction. Lower every allowed spin assignment through the existing typed
TensorIR algebra. Resolve the four mixed-spin amplitude permutations to signed
views of the single `t2ab` input. Keep both same-spin antisymmetries separate.

Use the full occupied/virtual Fock matrices in the residual graph, with no
denominator subtraction, division, damping, or level shift. CPU and CUDA
executors and JVP/VJP generation consume exactly this one mathematical graph.
Hamiltonian inputs are antisymmetrized physicists' integrals, not restricted
spatial Coulomb tensors.

## Rejected alternatives

- Reuse the restricted inventory with adjusted prefactors: its spin-free
  symmetries and normalization cannot represent the unrestricted coordinates.
- Embed all orbitals/amplitudes in dense combined-spin tensors in production:
  this unnecessarily retains forbidden spin blocks and duplicates storage.
- Evaluate `exp(-T) H exp(T)` in determinant space as the production algorithm:
  this is a valuable independent small-system oracle, not a scalable solver.
- Write CUDA equations separately or use PySCF as the production evaluator:
  either choice violates shared scientific ownership or the oracle boundary.

## Invariants

- Five stored amplitude blocks, independent oa/ob/va/vb, real-FP64.
- Physical projected residuals; canonical Jacobi comparison is `D*(new-old)`.
- Full-spin-orbital energy factors 1, 1/4, 1/2 before spin summation.
- Fixed-amplitude TensorIR cotangents are full-array duals, not automatically
  packed excitation-state Lambda duals.
- The compiler never imports PySCF, the public runtime, or a GPU runtime.
- Conventional four-index virtual inputs/intermediates remain explicit and
  budget-admitted; no tiled-provider or complete-endpoint performance claim.

## Evidence

`tests/python/test_uccsd_equations.py` compares all outputs with independent
fermionic determinant-space similarity transformations, PySCF UCCSD random
amplitudes and molecular solutions, and the RCCSD restricted limit. Both
native backends use the determinant oracle. JVP/VJP transpose and finite
difference gates exercise amplitude/Fock/integral differentiation. Empty,
one-spin and unequal-spin domains and fail-closed feed/budget admission are
covered. GPU qualification is run on n1 through a finite Slurm allocation;
source-matched output is retained locally under ignored `.artifacts/`.

## Consequences and remaining work

This is the equation foundation for #1821, not public UCCSD support or a complete
solver. Spin-resolved provider/reference admission, canonical denominators,
packed DIIS and iteration/convergence state still need implementation. #1822
needs independently normalized tiled unrestricted triples. #1824/#1825 need
solved Lambda, excitation-dual pullbacks, shared UHF orbital response, integral
derivatives and complete molecular force qualification. None of those issues
can be closed by fixed-amplitude graph tests.

## Revisit when

A spin-resolved provider needs a reduced canonical integral inventory, or a
bounded virtual contraction must be externalized without changing the physical
equations or generated Hamiltonian adjoints.

## References

- Issues #1821, #1822, #1824, #1825; amplitude-layout prerequisite PR #2234.
- `docs/developer/uccsd_equations.md` and `docs/developer/uccsd_layout.md`.
