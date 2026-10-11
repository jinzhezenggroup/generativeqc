# Decision: full-range hybrid Hessian planning by primitive rule

Status: implemented
Date: 2026-10-11

## Problem
MethodIR's stationary HVP planner registered semilocal rho/sigma only, even
though stationary-gradient codegen already supplied the full-range exchange
ordered-quartet weight and generated second-integral HVP providers existed.
Tests merely injected a hypothetical exchange rule.

## Decision
Register exactly one rule for ExactExchangePrimitive and reuse the existing
stationary-gradient full-range exchange objective/weights, TensorIR JVP and
bounded HVP contraction. The source requires both left/right density responses,
first-integral directions and weighted second-integral HVPs.

For an ordered quartet (i,k|j,l), the scalar coefficient is -a/4 times
D_ij D_kl in RKS (occupation-weighted D), or -a/2 times the same-spin
density-pair sum in UKS. Both density variations enter its JVP, without UKS
cross-spin exchange. The orbital response is the *hybrid* CPKS solution; it
cannot be reconstructed by adding separate HF and DFT molecular Hessians.

## Rejected alternatives
- Named PBE0/B3LYP Hessian drivers: duplicate generic exchange algebra.
- Promoting native/global-hybrid Hessians from compiled source algebra:
  unqualified hybrid CPKS, XC geometry, Pulay and complete source assembly.
- Differentiating SCF iteration tapes rather than the converged stationary
  implicit-response problem.

## Evidence and invariants
- PBE0/B3LYP with RKS/UKS: independent ordered-quartet displaced-gradient
  oracle at three steps; exact method fractions and same-spin weights;
  first-density response and weighted second-integral terms independently
  checked; TensorIR artifact replay checked.
- Custom exchange fractions and zero exchange keep name-free composition.
- Semilocal source identities remain unchanged. Range-separated, tau/meta,
  nonlocal and dispersion still fail closed. require_native_endpoint
  continues to reject both CPU and CUDA hybrid molecular Hessians.
- This PR claims no complete molecular or GPU endpoint performance result.
  Test status is determined by CI on the exact committed head.

## Follow-up
Reuse the method-neutral response/provider owner to qualify complete hybrid
CPKS, exchange RHS and Pulay relaxation, and full-source molecular Hessians.
Issue #1905 separately owns semilocal RKS CUDA residency.

References: #180, #1905, docs/developer/dft_hessian.md.

Agent: ChatGPT
Model: GPT-6
