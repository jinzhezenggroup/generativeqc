# Decision: stream independent CPU RHS schedules through the native response owner

Status: implemented
Date: 2026-10-16

## Problem

The public host GMRES bridge moved the inner controller to C++, but sequential
multi-RHS execution still had a Python admission/solve loop. Simply exposing the
existing retaining native batch would change public semantics: that batch is
all-or-nothing, while the Python endpoint preserves a successful prefix and
returns shared workspace-limit diagnostics for its tail. Retaining every native
result until Python has copied the whole block would also change simultaneous
live storage and duplicate diagnostics across the publication boundary.

## Decision

Add a streaming native sequence owner. Each admitted request uses the existing
problem-bound native response batch/GMRES implementation; no Krylov or physical
operator equations are copied. Publish borrowed views synchronously, release the
native result before the next request and charge actual earlier published
solution/basis sizes. A lease callback runs at request/publication boundaries,
including empty, zero-RHS and workspace-refused requests. Callback failures abort;
the summary is transactional, but already-executed callback side effects are not.

Borrow validated read-only RHS storage with checked positive element strides.
C/F-order and positive strided views need no whole-matrix conversion. Stage at
most one noncontiguous column, after capacity admission; reject invalid/overflowing
buffer extents without callback or dimension-sized allocation. The existing
GMRES workspace polynomial covers active result/publication-copy overlap;
earlier solution/basis copies and one staging vector are charged separately.
Publisher history copies/object bookkeeping and physical operator storage have
independent budgets. Do not call this a complete physical allocation receipt.

The Python adapter owns only dtype/layout admission, callbacks and detached
publication. Copy the scalar view header before its stack lifetime ends, detach
array data before return, preserve original Python exceptions and never unwind
them through C. Input arrays given to user operator callbacks must also be copies,
because callbacks and retained tracebacks may outlive the native vector lease.

## Boundaries and rejected alternatives

- Do not substitute an atomic whole-batch refusal for an admitted prefix.
- Do not convert every C-order RHS into a second column-major matrix just to
  satisfy contiguous native spans; its live storage would change admission.
- Do not skip lease validation because a zero RHS or resource refusal skips the
  physical action. Validation remains the method/provider owner's responsibility.
- Missing additive protocol, foreign dtype/negative/unaligned layouts and
  non-host domains keep explicit compatibility; native/load errors never retry
  Python science. Resident/device engines and user solution consumers are not
  routed through this CPU publisher.
- The value-only rank SVD, recycled/block scheduling, method RHS and seven-source
  molecular HVP assembly remain outstanding. This does not complete R3 or infer
  symmetry/equivalence between CPHF, CPKS, Lambda and nonsymmetric CC operators.

## Evidence and revisit conditions

Native/public-header-only tests cover independent nonsymmetric solutions,
strided/contiguous staging, exact/below-budget prefix behavior, empty and stale
leases, publication failure, forged plans, overflow and tracked result cleanup.
Python tests cover independent scaled dense systems, layout/preconditioner
combinations, callback exception identity and retained inputs, old-protocol and
foreign-domain compatibility. Complete endpoint comparisons use matched baseline
Python/library artifacts, actual residual/convergence/work signatures and the
same construction-through-cleanup timing scope as the host-controller slice.

Revisit the compatibility/rank boundaries only with independent numerical,
lifetime, failure, resource and complete-endpoint qualification of those domains.
See #2227, the public host GMRES controller decision and `docs/developer/response.md`.
