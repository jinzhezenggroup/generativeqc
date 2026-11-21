# Decision: Compose the r²SCAN-3c D4/EEQ Hessian by directional adjoints

Status: implemented (CUDA qualification build only)
Date: 2026-11-20

## Decision

Reuse the augmented symmetric EEQ charge system `A x = b`, including its
fixed-total-charge constraint. Factor `A` once per direction and solve
`x_dot = A⁻¹(b_dot − A_dot x)`. The fixed-charge BJ/ATM evaluator propagates
position and charge tangents through coordination, reference weights, C6,
damping, and switching. It supplies the explicit geometry gradient and its
tangent together with the charge seed `p = E_q` and its total tangent.
The same LU solves `λ = A⁻ᵀ p` and
`λ_dot = A⁻ᵀ(p_dot − A_dotᵀ λ)`. Differentiating the pairwise contraction
`λᵀ(db − dA x)` adds the implicit gradient action without building a Cartesian
charge Jacobian or a third-order tensor.

The named preparation owner is authoritative for the correction profile and
geometry. The Hessian borrows the active D4 and gCP owners only under the
matching live SCF token. It stages their sum with the electronic action on
the producer stream, checks finiteness and both correction statuses, and
then publishes. An active correction without a matching second-order owner
refuses preparation instead of yielding an electronic-only result.

## Evidence and limits

- `tests/native/test_d4_eeq.cpp` checks the complete HVP against independently
  recomputed first-order gradients, with a maximum directional error of
  `6.384e-13` Hartree/Bohr² in the qualification run.
- `tests/native/test_d4_eeq_cuda.cu` checks the real-device directional action
  and capacity/failure isolation. The named molecular test separately checks
  the corrected energy and gradient, raw matrix, HVP and translations. All
  four RKS/UKS × Direct/DF hydrogen fixtures passed the independently
  reconverged Libxc/PySCF + D4/gCP comparison, with maximum matrix error
  `1.32e-8` Hartree/Bohr². The 34-AO water fixture additionally checks the
  fitted HVP against displaced native forces; it is not an independent
  absolute curvature reference.
- Dense EEQ factors and D4 response data have bounded quadratic storage. The
  initial complete D4 CUDA evaluator runs serially in a device kernel; the
  bounded schedule is a correctness baseline, not a speedup. A production
  optimization needs complete HVP/Hessian timings and semantic work counts,
  including the ATM term and the repeated directional solves.

## Fitted metric prerequisite

The 34-AO water fixture exposed cancellation when the Coulomb metric
cotangent was lowered as `-M⁻¹ (q qᵀ / 2) M⁻¹`. The prepared cotangent
failed its raw symmetry audit by up to `4.51e-11`. Reuse the same fitted
charge `z = M⁻¹q` on both legs of `-z zᵀ / 2`, and differentiate that
factorization. This exploits the symmetric physical inverse `WᵀW` rather
than repairing a published Hessian or relaxing any acceptance tolerance.
The conditioned-metric test compares both the weight and its tangent to
independent solves. Exchange retains its separate cotangent.
