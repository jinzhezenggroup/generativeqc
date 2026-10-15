# Decision: share the native CPU GMRES controller through a public callback ABI

Status: implemented
Date: 2026-10-16

## Problem

Native method consumers already use `response::solve_gmres`, but installed Python
host response solves still duplicated Arnoldi, least-squares and residual control.
Moving those equations into another adapter would preserve two scientific owners.

## Decision

Expose the existing C++ controller through an additive public C ABI and move-only
C++ result owner. The Python adapter marshals requests and publications; physical
operators, reference/lease validation and method-specific equations stay separate.
No symmetry is inferred: nonsymmetric CC, CPHF, CPKS and Lambda are not interchangeable.

Admit positive-dimensional host solves without a foreign vector engine, declared
device storage, solution consumer or general preconditioner. Existing diagonal
preconditioning remains opt-in. Sequential/recycled inner solves use this path;
blocked solves and inter-RHS/recycle schedules do not migrate in this slice.
Missing optional protocol preserves compatibility, but native rejection or a
selected library's load failure never authorizes a Python numerical retry.

## Numerical and lifetime invariants

- Accept using the true unpreconditioned residual. Native Givens/triangular
  arithmetic, cached preconditioned columns and predicted-residual checkpoints
  can differ from Python least squares in work counts and iteration histories.
  Compare numerical gates and report actual work rather than require exact history.
- Admit capacity before dimension-sized native allocation or callback work.
  Report tracked solver numeric storage, not callback/provider or whole-endpoint
  physical allocations. Wrapper snapshots and publication copies remain separate.
  Sequential native batch plans must also charge prior results' reserved history
  and basis capacities when diagnostics are enabled, rather than assuming that
  only solution vectors survive between requests.
- Copy callback inputs before invoking Python: callbacks and retained exception
  tracebacks can otherwise keep NumPy views into freed native vectors. Catch
  exceptions inside ctypes, destroy native publications and rethrow the original
  Python exception; never unwind through C.
- Native result views borrow immutable arrays until owner destruction. Inputs,
  callback objects and operator state need not survive synchronous completion.

## Rejected alternatives

Do not copy the GMRES implementation, substitute a dense reference solver, infer
operator symmetry, route declared device providers through CPU, or claim that
controller ownership establishes a fully native molecular Hessian/HVP endpoint.
Retire compatibility only after its distinct residency/consumer/operator domains
have equivalent numerical, failure, lifetime, resource and endpoint qualification.

## Evidence and revisit conditions

The public-header-only SDK test covers a nonsymmetric independently solved system,
workspace admission, transactional callback failures, nonfinite actions, borrowed
views and move ownership. Python tests cover scaled independent dense solutions,
exception identity, retained callback vectors, failure cleanup, device exclusion
and sequential/recycled solves. Native RKS/UKS suites supply independent
libcint/Libxc actions and reconverged-density checks. The opt-in complete CPKS
benchmark records construction, SCF, response preparation, solves and cleanup,
with exact Python/library provenance and semantic work for parent comparison.

Revisit the eligibility boundary after actual device/resident and solution-consumer
contracts are qualified. Full R3 completion still requires method-specific RHS,
multi-RHS/recycle and molecular HVP assembly, not merely this controller.

References: #2227; `docs/developer/response.md`; `docs/reference/native_api.md`.
