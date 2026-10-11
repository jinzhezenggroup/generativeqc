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

Endpoint admission is bounded by the existing consumer contract. CUDA KS force
descriptors admit s/p/d only: retain through-f and def2-TZVP energy/D/F diagnostics
as negative complete-force evidence, and qualify through-d plus def2-SVP endpoints
independently. Unsupported force rows cannot earn profitability or authorize a
CPU/PySCF force substitution. Do not change the original f fixtures into d fixtures.

The first native HF endpoint fixture omitted the molecular owner's radial and
contraction normalization. Both generated/library runs then agreed with each
other but failed the independent reference. Normalize raw shells using the same
public molecular preparation seam before creating the cached owner. Separately,
building the main library does not rebuild optional stationary force companions;
their old compiler contracts correctly failed closed. Build the companion manifest
targets and retain the rejected rows, rather than relabeling old binaries.

## Complete-endpoint retry and value-schedule evidence

The first matched endpoint sweep exposed a cold HF lifetime error: the arena
already reserved dense basis publication scratch, but its pointer was resolved
before the immutable structured binding existed. The initial library capture
therefore failed and the established retry rebuilt the owner with generated
kernels. Resolve the span after binding preparation. Scientific agreement alone
did not reveal this loss of the candidate; successful final binding diagnostics
and actual submission records did.

An abandoned cuBLAS capture must not be authenticated by a later generated
Graph launch. Endpoint reports now require a successful HF row to retain the
same operation's cuBLAS binding before crediting stream/captured library work.
Keep failed attempts as diagnostics, not positive adoption evidence.

Semilocal KS can use its independent MD-J value schedule, which does not submit
the four public/direct basis stages. The explicit diagnostic `--disable-md-j`
ablation exercises the shared Direct value owner without changing its default.
Record provider controls in timing identities and retain default-schedule and
MD-disabled evidence separately; never compare their costs as a provider-only
speedup. Error receipts retain the same frozen provenance as successful rows.

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
