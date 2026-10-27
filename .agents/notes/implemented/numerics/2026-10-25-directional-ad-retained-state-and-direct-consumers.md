# Decision: directional molecular AD, retained HF states, and direct consumers

Status: implemented
Date: 2026-10-25

## Problem

The integral AD chain already differentiated normalized Cartesian RHF/UHF
integrals and relaxed orbital response. The remaining application bottlenecks
were its frontend composition and execution schedule: Array API second-order
sources materialized full joint Hessians, separate directional actions repeated
primal SCF, and dense ordered ERIs and stationary weights limited capacity.
PyTorch rejected higher-order backward even for qualified native contexts and
did not expose joint Gaussian parameter derivatives.

This extends the scientific contract documented in
[the spin-response integral AD decision](2026-10-23-spin-response-integral-autodiff.md).
It does not add new electronic-structure response equations.

## Decision

`IntegralHF.prepare` returns an independently owned stationary snapshot. Native
ownership includes copied system/parameters, integral provider, converged spin
densities, physical Fock matrices and energy-weighted densities. Response actions
reuse this state without re-solving primal SCF. The compatibility v2 ABI remains
a transactional wrapper around the same implementation. Separate snapshots are
needed for concurrent execution; close and execution must be serialized.

Array API linearizes first-gradient source bindings into directional gradient
bindings. One physical HVP supplies the latter; the surrounding tensor AD
propagates both outer-loss curvature and coordinate/parameter transformations.
Disconnected source gradients are removed. The integral path shares a retained
state for energy, gradient and response. Other native methods use their existing
context-qualified HVP and may solve SCF again; no general reuse claim is made.

PyTorch native energy and ragged batches bind a differentiable gradient to the
qualified native HVP. Calculator identity is captured at energy forward and
checked before creating and executing second backward. `integral_energy`
supplies joint coordinate/exponent/coefficient gradients and double backward;
its graph owns the retained state even after the public owner closes. Torch's
saved-input versioning rejects in-place mutations. Third derivatives and Torch
transform/vectorized-backward protocols are not implemented.

The optional CPU `integral_storage="direct"` schedule passes primitive values
directly to generated J/K and contracts generated stationary source weights
before the integral pullback. Scientific weights reuse the existing compiler
IR/scalar algebra; native callbacks own execution, snapshots and budgets only.
No Python callback is used inside a Fock/response action. Runtime templates are
included in installed compiler assets and generated-artifact identity.

Explicit device-qualified consumer members also bring `parameter_ad_runtime.hpp`
into the CUDA ownership census. Its new shard conservatively classifies the
existing primitive contraction/product-rule and mixed execution region as
scientific, with staging and ABI wrappers as runtime. The census adds 153
scientific and 85 runtime lines because the shared header was previously absent;
the physical header changes from 211 to 238 code lines. This is not 153 lines
of newly introduced CUDA mathematics or a retirement of the shared contraction.
The shard records independent numerical gates required before moving that
composition into generated contraction lowering.

Direct numeric storage is quadratic in AO count plus parameters/tape scratch,
but ordered primitive work remains quartic and repeats for each J/K action.
The complete HVP charges primal, any shared first-gradient source, and response
work to one `max_work` allowance. Full Hessian columns consume one shared
remaining allowance. Native
Fock callbacks enforce the admitted remaining work before each source action,
so rejection does not require first completing an over-budget endpoint.
Successful diagnostics accumulate semantic work; failures preserve the previous
successful receipt and permit retry on the same physical state.

## Boundaries and rejected alternatives

- Dense remains the default and the supported CUDA integral storage route.
  Direct is explicit: it trades recomputation for memory, without an automatic
  backend switch or a performance promise.
- Explicit integral values/JVP/VJP still publish ordered dense tensors and must
  pass a separate capacity gate, even for a direct owner.
- A full Hessian is an explicit column assembly with one primal state. It is
  not a prerequisite for an HVP and is neither projected nor symmetrized.
- Numerical differences and PySCF are acceptance oracles, never production
  derivatives or optimization dependencies. The example's `--reference` flag
  runs only independent comparisons of the actual raw basis.
- Arbitrary nested AD, third derivatives, masked molecular second order, DF
  parameter AD and KS/post-HF basis-parameter AD remain outside this contract.
  Supporting them requires their own qualified derivative consumers.

## Evidence

Qualification starts from master `3ef7d7559ebd5ec941c76584caa7706c4941e6ee`
in branch `codex/autodiff-completion-20261025`. Local evidence is retained under
ignored `.artifacts/qualification/` in the task worktree.

- The broad CPU regression passed 430 tests and 17 subtests, with eight
  explicitly GPU-only skips. Three additional application/native-frontend
  acceptance tests passed. Follow-up suites passed 169 frontend tests, 46
  integral/application tests, and three focused complete-budget tests after
  configuration-identity and direct-budget hardening; these overlapping suites
  are not additive test counts.
- RHF/UHF integral derivatives retain independent PySCF/libcint gates. A 16-AO
  H4 s/p case completes direct energy, gradient and relaxed HVP within a 2 MiB
  numeric-array allowance that rejects the dense owner. Independent gates are
  3e-9 Hartree for energy, 2e-8 for coordinates and 3e-8 for nuclear HVP.
- Torch synthetic curvature tests compare complete nonlinear outer losses to
  independent quadratic algebra, including gradgradcheck. Joint raw-basis
  curvature uses separately converged PySCF directional energies. Real native
  LDA composition is checked in both public frontends.
- The water/STO-3G basis optimization example lowered energy from
  -74.96308295573917 to -75.0490886217103 Hartree in three accepted steps.
  Initial/final independent raw-basis errors were 1.71e-13 and 4.41e-13 Hartree.
  These energies demonstrate the workflow, not transferability of the basis.
- One H2/STO-3G CPU FP64 endpoint sample, with an already-created owner and
  shared artifact cache, took 0.00815 s for one HVP (one SCF, one response) and
  0.11300 s for its 18-column joint Hessian (one SCF, 18 responses). Three
  retained directions took 0.01730 s (one SCF, three responses); three fresh
  calls took 0.01972 s (three SCFs, three responses). H@v agreed with HVP within
  3e-10. These are bounded small-case timings, not large-basis scaling claims.
- A wheel-layout projection was executed outside the checkout: installed dense
  and direct energy/gradient/HVP agree, with no source-tree asset fallback.
  Compiler/SCF structure, public contracts, native boundaries and Sphinx with
  warnings treated as errors passed.
- All seven real-device integral AD gates passed on n1 (Slurm node1, main
  partition, RTX 5090) using a finite `srun` allocation. These include RHF/UHF
  integral/JVP/VJP/HVP parity, p-shell independent reference, Array API joint
  HVP, error recovery, and continued state use after its original owner closes.
  Remote build and qualification logs are retained under
  `n1:/data/jzzeng/qc-autodiff-completion-20261025/.artifacts/qualification/`.
- Compute Sanitizer memcheck with full leak checking passed the two complete
  RHF/UHF CUDA gates, including retained-state lifetime: zero errors and zero
  leaked bytes. The sanitizer also ran inside a finite Slurm GPU allocation.

## Revisit when

A screened/tiled schedule can bound complete endpoint work while differentiating
the same scientific approximation, or another backend admits direct generated
J/K/response consumers with independent numerical and real-device qualification.
Extending native retained-state reuse requires method-specific lifetime and
response contracts; it should not be inferred from frontend API compatibility.
