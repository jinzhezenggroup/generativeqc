# Decision: source-matched structured SCF endpoint admission

Status: implemented
Date: 2026-10-11

## Problem

Primitive reconstruction receipts do not qualify a complete scientific solver.
KS Direct sources also outlive the method matrix owner, and spherical public AO
dimensions are not the physical Cartesian contraction dimensions.

## Decision

Extend the prepared Fock submission seam with a call-local borrowed matrix view.
KS explicitly prepares public/direct dimensions and spin counts. Generated J/K
owners consume shared basis stages without retaining method pointers or creating
another handle. Density stages borrow dead contraction outputs; projection stages
borrow dead transformed density. The latter must never be reused to implement a
second RSH contraction without reconstructing it. Therefore existing fused RSH
submissions without a borrowed view retain their original native implementation.

Charge/upload shell-local spans for generated sources. They measure actual
reduction envelopes, not coefficient sparsity. Retain native sparse execution;
only diagnostic preparation may request the dense ablation. A test-only semantic
region mask freezes library offers before graph capture and never repins an
existing binding. Resource preparation uses a separate fact from lifetime-stable
binding workspace use; a zero-length arena region still has a nonzero offset and
must not be mistaken for allocated capacity.

## Acceptance

Use complete cold/warm/moved RHF/UHF/RKS/UKS endpoints, detached D/F diagnostics,
generalized-eigen/metric/final-state gates and pinned independent PySCF/Libxc.
KS finite differences must reconverge the explicit moving quadrature, at two
steps. Diagnostic copies and intrusive tracing do not supply clean endpoint
timing. Failed rows survive; unknown profitability remains unknown. Library
defaults stay closed until scientific and matched profitability evidence passes.

Matched profitability requires three independent process pairs, never three
members of one batch. Identity includes source, library, scientific inputs,
Graph mode and GPU environment. Cold KS cost includes preparation. Keep separate
warm, geometry-refresh and moved-warm gates and retain iteration work counts.
The report cannot promote defaults and does not equate preparation with execution.

The independent oracle uses a 1e-9 orbital-gradient convergence target and 1e-12
energy tolerance. On the 96-AO UHF diagnostic, a stricter 1e-10 iterative target
oscillated at numerical noise and reported nonconvergence even after independent
Newton retries. Those rejected receipts remain historical failures. The 1e-9
target converges independently; physical residual and all endpoint acceptance
thresholds remain unchanged. Native HF orbital downloads are column-major,
unlike exported symmetric matrices; transpose each orbital frame before applying
generalized-eigen/metric gates rather than weakening those gates.

## Rejected alternatives

- Store a KS binding in the integral source: risks dangling views after geometry
  replacement or method destruction.
- Allocate another full basis scratch plane: existing dead source planes suffice.
- Replace determinant-consistent HF Pulay weights by canonical weighted Gram:
  changes finite-residual force semantics; retain P F(P) P and its spin factors.
- Promote from primitive receipts: misses solver convergence and endpoint costs.

## References

- PR #2263; issues #1877, #1878 and #1883.
- `2026-10-11-borrowed-structured-scf-regions.md` records the primitive foundation.
- `tests/native/structured_scf_endpoint_probe.cpp` and
  `benchmarks/structured_scf_endpoint_qualification.py` are runnable acceptance
  paths; evidence records source and actual loaded library identity.
