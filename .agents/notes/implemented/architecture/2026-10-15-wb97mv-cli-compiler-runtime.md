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

The numerical/native candidate is
`80dfd1bf08f4051c94f883c53896b294d55b3ff1`, based directly on master
`540ec82011232606374747727cd30fd7e1f93970`. The PR excludes the original
worktree's unrelated dirty files and unmerged parent-branch changes. Master had
moved KS spin admission into `dft_method_preparation.cpp`; the same WB97M-V-only
ECP qualification exception is applied at that current owner.

- A complete clean Release CUDA-sm120 build and a subsequent candidate-identity
  rebuild use verified ccache 4.5.1. Generated native compiler commands invoke
  that launcher; the cache is reused, not cleared. Shared counters are not
  interpreted as an isolated hit-rate benchmark.
- All 17 native CLI CTest entries pass in 142.13 seconds, retaining ordinary
  Python-free runtime, geometry, Gaussian/DF, profile and catalog contracts.
- The final expanded Hessian CLI suite runs ten tests in 42.273 seconds: nine
  pass and the monolithic-only missing-library gate is skipped in this dynamic
  build. It checks raw CPU H@v parity, an independent reconverged native-force
  oracle, explicit interpreter/library selection, invalid shape/provider
  rejection, worker failure/timeout and no partial publication.
- Compiler/TensorIR/grid/native-launcher regression: 153 pass and two opt-in
  GPU tests skip in 20.83 seconds. The current master compiler inventory contains
  548 modules with zero dependency errors. Owned Ruff, C++ clang-format and
  patch-whitespace checks pass.
- Slurm node1 RTX 5090 job 8657 directly invokes the compiled CLI's CUDA HVP
  command (not a Python API substitution). Its JSON reports device residency,
  zero numerical host fallbacks and a finite complete result.
- Final-source Slurm job 8655 passes all 24 GPU tests in 1032.50 seconds: six
  complete endpoints, sixteen resident lowering/response/failure gates and two
  compiled-CLI endpoints. The latter include raw RKS matrix/HVP publication and
  combined UKS/DF/ECP with a genuinely spherical d auxiliary record and nonempty
  response spaces in both spins. The scheduler-assigned visibility is preserved;
  no work runs on the maintenance node. Slurm accounting storage is disabled,
  so the retained srun/pytest log supplies the job ID and successful exit receipt.
- The clean candidate's 32 CPU Hessian/basis/domain tests all pass in 1132.35
  seconds, including asymmetric s/p water, coupled UKS, DF and ECP independent
  force/energy oracles. Together with the earlier 153 tests, 185 Python tests
  pass, with the two explicitly opt-in GPU tests skipped in that CPU invocation.

The qualified native CLI SHA-256 is
`6cd50d355dbb9278711985fbf3be4b1ea4228e1a808bb17c8dea4227f1ea9af6`;
its source-matched shared library SHA-256 is
`2af4e1b367ec2b0058848f33321582eab5c30ddbaf33e6d05c39ad0e9e2058c8`.
These are one-off artifact provenance, not permanent source-fingerprint tests.
Logs start with `/tmp/wb97mv-cli-`; final receipts are also retained in the
ignored `.artifacts/wb97mv-hessian-cli-20261015/` directory.

The resident numerical qualification is preserved in
[the numerical ownership note](../numerics/2026-10-15-wb97mv-resident-hessian.md).

## Revisit when

A generated standalone native execution ABI for the complete stationary method
is available. Replace only the worker boundary, retaining compiler ownership,
admission gates, native provenance, no-fallback CUDA execution and independent
whole-endpoint numerical qualification.
