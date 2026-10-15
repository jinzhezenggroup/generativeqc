# Decision: explicit compiler-runtime Hessian commands in the C++ CLI

Status: implemented
Date: 2026-10-15

## Problem

The complete WB97M-V compiler-AD Hessian/HVP was available only through Python.
The user requires C++ CLI access, while ordinary native CLI commands retain a
Python-free runtime contract. The current generated native KsComposition CLI
catalog does not itself supply the complete nonlocal stationary consumer.

## Decision

Add explicit `hessian` and `hvp` subcommands to the existing C++ executable.
They require an absolute `--compiler-python` executable path and call a private,
versioned compiler-runtime worker. Do not imply that the new numerical endpoint
is Python-free. Ordinary `run`, `resources` and discovery remain unchanged.
The existing compiler owns all differentiation and the existing native library
owns SCF. No second C++ scientific algebra, finite difference or host derivative
fallback is introduced to make a CLI flag appear native.

The C++ parser reuses common run validation but does not use KsComposition's
nonlocal rejection as the admission policy for the distinct compiler consumer.
The extension admits only canonical WB97M-V RKS/UKS and FP64, and the original
endpoint retains all resource, rank, quadrature, response and backend gates.
Every accepted flag is forwarded; unrelated method/resource/force flags fail.

POSIX spawn uses an argv vector and inherited Slurm device visibility, never a
shell or PATH interpreter lookup. Private unnamed files avoid pipe deadlock and
leave no geometry or result files on failure. The child process group has a
finite timeout, including JIT descendants. The worker loads the exact library
identified by `dladdr` in the parent, overriding an unrelated Python-library
environment selection, and returns its native source identity and resolved path.
The C++ owner checks those, the operation/method/backend, units, finite shape,
compiler-AD provenance and CUDA no-fallback residency before any publication.
Only input metadata and the final numerical result cross this process boundary;
CUDA intermediate panels stay inside the existing resident owner.

The protocol has a separate explicit 64 MiB request/result cap. This is a CLI
serialization limit, not a redefinition of the existing scientific memory
budgets. Host preparation and final JSON serialization remain host work.

The current master also supports a monolithic static CLI. That endpoint requires
an explicit absolute `--compiler-library` shared library from the same source
build, checked against the native source identity before SCF. A dynamic CLI
rejects an override that differs from the library already linked into its process.

## Rejected alternatives

- An implicit Python console-script fallback would break the native CLI contract
  and could select an unrelated package/library/interpreter from PATH.
- Reimplementing Hessian mathematics in C++ would introduce a second scientific
  owner instead of consuming the already qualified compiler-AD endpoint.
- Silently admitting WB97M-V through ordinary native `run` would pretend that
  its different KsComposition-only execution contract implements this consumer.

## Evidence

Qualification results and source-matched logs are appended after execution.
The resident numerical qualification is preserved in
[the numerical ownership note](../numerics/2026-10-15-wb97mv-resident-hessian.md).

## Revisit when

A generated standalone native execution ABI for the complete stationary method
is available. Replace only the worker boundary, retaining compiler ownership,
admission gates, native provenance, no-fallback CUDA execution and independent
whole-endpoint numerical qualification.
