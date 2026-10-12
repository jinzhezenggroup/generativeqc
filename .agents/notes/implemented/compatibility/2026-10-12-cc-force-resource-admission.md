# Decision: replace conventional CC force AO ceilings with resource admission

Status: implemented
Date: 2026-10-12

## Problem

The public RCCSD and RCCSD(T) force owners rejected all requests above 12 and 56
public AOs respectively before consulting their existing complete-force resource
planner. These counts were historical qualification frontiers, not dimensional
restrictions of the analytical force equations. They also prevented callers with
sufficient memory from attempting a calculation and hid actual resource failures.

## Decision

Remove the fixed dimension guards from both public prepared owners and the
shared `plan_relaxed_rccsd_force_cpu` / `relaxed_rccsd_force_impl` paths. Retain
nonempty occupied/virtual spaces, matched physical references and AO dimensions,
checked size arithmetic, complete simultaneous host/device capacity accounting,
CPU/CUDA backend ownership, the bounded derivative provider, and every numerical
convergence, canonicalization, curvature, residual and stationarity gate.

Do not replace the removed limit with a default opt-in or a larger arbitrary
constant. Requests that cannot fit the real resource plan fail with a resource
error; numerical failures remain numerical failures. This is a change in admission
policy, not a new large-basis algorithm or precision policy.

## Alternatives rejected

- Raising both ceilings to 56/112 AOs would merely postpone the same problem.
- Automatically changing conventional CC to density fitting would alter the
  interaction Hamiltonian, so DF stays an explicit separate method.
- Removing numerical stability checks to admit difficult orbitals would risk
  silent incorrect gradients.

## Evidence and remaining qualification

The new planner contract probe covers 13 and 57 AO dimensions for both CCSD and
CCSD(T) across CPU/CUDA planning, with high-budget admission and tight-budget
refusal. The public API regression exercises a 112-AO request under a deliberately
insufficient budget, requiring the resource-error path rather than `NOT_IMPLEMENTED`.
Existing independent small-system forces and the 14/28/56-AO CCSD(T) reference
frontier remain the scientific regression suite. These tests do **not** establish
independent large-scale CCSD or beyond-56-AO CCSD(T) gradient accuracy. Extended
PySCF force/finite-difference, resource-peak and real-device qualification is
still required before publishing a larger validated frontier.

## Consequences

Dense MO ERIs, response matrices and generated intermediates can become expensive
at larger dimensions even if a job has the required budget. The scheduler/resource
contract and complete endpoint measurements remain authoritative; the removed
integer checks were not a performance guarantee.

## References

- `src/cc/rccsdt_force.cpp` and `src/methods/rccsd_method.cpp`
- `tests/python/test_cc_force_cuda_provider_admission.py`
- `tests/python/test_rccsdt_public.py`
- `docs/developer/rccsd_t.md`
- `.agents/notes/implemented/performance/2026-10-03-cc-larger-force-work.md`
