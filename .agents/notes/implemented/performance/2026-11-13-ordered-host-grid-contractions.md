# Decision: ordered SIMD execution of shared host grid contractions

Status: implemented; component-qualified, not complete endpoint acceptance
Date: 2026-11-13

## Diagnosis

The sealed `host-nuclear-xc-phase-profile-20261113` experiment uses the exact
saved native 81-AO, six-atom, 87458-point v4 grid with no SCF, JK, W or response
solve. Its arithmetic-identical namespace copies reconcile complete component
work and produce identical output bits. Generated tensor contractions, not
scalar XC, dominate both nuclear XC consumers:

- Fock direction: 15.6065 seconds overall; feature, feature-direction and
  potential-direction maps take 1.9489, 5.7725 and 6.6670 seconds.
- Fixed-D curvature column: 36.9822 seconds overall; features, first-relaxed,
  pullback and relaxed pullback-direction maps take 1.9572, 5.7972, 9.8039 and
  13.8204 seconds. Scalar closures take 0.6989 seconds.

This is not evidence that every complete Hessian phase has the same bottleneck.
The phase archive SHA256 is
`aa7dfa067cbd0a0e253b4e5e8580c68b38207e0e224793b03d09abcea0fa6810`.

## Decision and ownership

Keep the previously qualified shared TensorIR graphs, including AD/CSE and all
primal validation. Add a default-false, explicitly boolean ordered-binary
schedule to `tensor/host_affine.py`; only the two native grid-map generators
opt in. The compiler classifies rank-two binary contractions with one shared
reduced label and an exact unit coefficient, then supplies dense operand strides.
The native execution template belongs to
`src/tensor/ordered_binary_contraction.hpp`, not the scientific compiler.

Runtime AVX2 detection selects four independent output columns. Each entry
retains operand order, multiply/add grouping and the serial reduced-label
sequence. Loop-invariant broadcast/contiguous/strided loads are specialized
outside reductions. There is no horizontal sum, FMA, BLAS, numeric allocation,
packing panel, changed mathematical work or additional admitted arena. Empty
output dimensions return before traversing another extent; empty reductions
write positive zeros without reading operands. Vector tails remain scalar.

Nonunit coefficients, unsupported node shapes and unsupported CPUs retain the
original generated scalar code. Nonunit fallback matters: moving a coefficient
across a product can change rounding, underflow and intermediate overflow even
when the final real-valued product is representable. All original input/layout,
alias/alignment, checked-size, arena and intermediate-finiteness guards remain.
Public outputs publish only after the complete graph succeeds.

## Evidence and rejected alternatives

The production-source component comparison binds source
`26f199f94962c7da7faeb4d5a7c642b19a8e803d47a584ed3bfcc782f7f9ab34`
and core SHA256
`78c5349dd295a252c1a6ce3f4555adebc943f130286d37594c674d20eb7bf0c9`.
It reconstructs the same D from the original native producer's saved C, retaining
that older producer identity separately; it does not invent a new stationary
state or run a reference/native SCF. Both complete component outputs are bitwise
equal to the same shared graphs with original scalar execution:

- Fock direction: 15.599559380 to 7.571075006 seconds;
  14647247478 map terms, unchanged.
- Fixed-D curvature column: 36.953800751 to 17.678803861 seconds;
  33556112408 map terms, unchanged.

The generic unit-stride prototype was correct on these components but slower
than the specialized load modes. Its archive SHA256 is
`c01d22c7191b0e27da93f6df327ec7261a94e59e8d8c5f1b44a69b7533b85451`.
The specialized prototype archive SHA256 is
`2fe851a3929de4b4138ff22253d59f304ef2d9ede99d7b97e5bf7d5e9c1c2862`.
Failed namespace/include harness attempts are retained rather than restamped.
Naive BLAS remains disqualified by its previously measured vacuum-branch changes;
this decision does not reopen it or alter a scalar XC branch.

Permanent compiled tests compare the same shared graph's output bits, work and
arena queries in native and forced-scalar modes, through 81 AOs/257 points.
Separate direct unreassociated graphs and scalar stencils remain independent
numerical controls. Signed nonsymmetric D/D', cotangents, zero reductions,
subnormals/signed zero, extreme exponents, nonunit-coefficient overflow,
near-vacuum classification, alias/capacity/refusal transactionality and all nine
native stride combinations are exercised. ASan/UBSan covers 216 native cases,
including huge empty shapes and vector tails. Physical AO/Libxc and actual
coupled RKS/UKS controls retain independent displaced-primal/SCF oracles and
nonempty density response. Raw commands and individual outcomes belong to
`build/qualification/host-nuclear-ordered-source-20261113`.

## Scope and revisit conditions

These timings are two components, with empty/frozen D_right on the saved-state
fixture, not full response, raw-Hessian, modes or optimizer speedups. The
independent nonempty-D_right tests do not turn that fixture into an endpoint.
The old 1800-second whole endpoint is still negative; source preparation,
stationary gradient, frozen integrals and coupled column work need their own
complete costs and acceptance. Both public ECP saddle guards remain closed.

Do not widen scheduling eligibility, reassociate reductions or change primitive
FP semantics without independent numerical, exact same-graph boundary and
complete-component qualification. A future ISA or a better contraction schedule
may replace this template only with the same bounded portable fallback and
scientific/endpoint gates.

## Follow-up

The [row-invariant register-reuse decision](2026-11-14-row-reuse-in-ordered-host-contractions.md)
extends this ordered execution schedule without changing any per-output
reduction sequence, mathematical work or numeric storage requirement. Its
complete-component qualification is separate from the historical measurements
above and does not qualify a full physical Hessian or optimizer.
