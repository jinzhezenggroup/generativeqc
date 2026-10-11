# Decision: compile user Array API primal and derivative programs natively

Status: implemented
Date: 2026-10-11

## Problem and decision

CPU eager conformance did not deliver the user's quantum-chemistry compiler
contract. Make native CPU the experimental compile default, retain explicit
reference execution, and connect explicit CUDA to existing TensorIR planning
and execution. Public grad/value_and_grad/JVP/VJP generate canonical derivative
Programs rather than invoking reference AD. Combined primal/reverse graphs
share SSA nodes and ordinary backend preparation. Explicit TensorSpecs preserve
AO/orbital/spin identities, never inferred from equal axis extents.

Native CPU remains float64 and gains canonical exp/log/sqrt/power with the same
finite/domain gates as reference/CUDA. Byte/work/node budgets and finite compiler
processes apply to primal and derivative endpoints. There is no implicit dtype
promotion or backend fallback. CUDA stages host inputs explicitly, not through a
GPU eager namespace. Pullbacks snapshot forward inputs and expose owner cleanup.

## Rejected alternatives and invariants

- Reference AD hidden behind grad would disguise oracle work as compilation.
- A second tape/rule registry would duplicate canonical scientific AD ownership.
- Generic shape convenience cannot erase explicit scientific index domains.
- Eager admission never authorizes unsupported native/derivative operations.

## Evidence and limits

Native tests disable reference execution/AD, compare RHF energies and density
derivatives to independent Coulomb/exchange/Fock contractions and finite
differences, and protect nonlinear derivatives, JVP/VJP, mutation snapshots,
cache reuse and resource rejection before toolchain discovery. Reference preview
tests now explicitly select reference instead of claiming native bool/float32.

Local Slurm discovery shows only node3, outside the allowed n1/n2/n4/n5 list;
no GPU was used. Planner/source checks and opt-in scheduled-device tests are
distinct from real-GPU qualification. Higher-order public transforms, dynamic
control flow, integer/complex compilation and full linalg/fft remain unsupported.

This supersedes execution, not eager conformance, in
[the eager boundary note](2026-10-11-array-api-standard-eager-boundary.md).
Revisit when additional primitives/backends and higher-order transforms have
independent gates, or allowed Slurm GPUs provide real-device evidence.
