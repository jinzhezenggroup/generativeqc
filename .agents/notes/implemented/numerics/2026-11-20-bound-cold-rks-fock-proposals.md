# Decision: Bound cold B3LYP RKS orbital proposal mixing

Status: implemented
Date: 2026-11-20

## Problem

The screened Cs46 first-order candidate had a large physical gap at the
independent root, yet native Hcore-start SCF exhausted the same 240-iteration
budget at five different DIIS histories. Increasing the DIIS history neither
tracked the correct root nor diagnosed a derivative error. See the
[isolated prequalification](../../proposed/2026-11-20-precondition-cold-cs-scf-with-bounded-fock-damping.md).

## Decision

Pass an optional fixed cold proposal prefix through an additive named KS
batch C ABI. The existing ABI and physical MethodIR do not carry this SCF
policy. Admit one CPU direct-FP64 B3LYP-D3(BJ) RKS item; refuse an unsupported
provider, preliminary guess, or insufficient separately reserved AO-square
workspace before preparing numerical owners. Never silently retry a refused
prefix without it. The first proposal is unmixed; subsequent prefix steps
mix with the previous proposal while DIIS waits. All prefix iterations spend
the original solver budget. Only diagonalization sees the proposal. Energy,
physical commutator, final full Fock rebuilds and derivative-state publication
retain their ordinary semantics. Do not apply the policy to a warm frame or
warm fallback. The explicit saddle options attach only before the first
accepted derivative frame, with accepted-frame warm starts and root tracking
thereafter. Python requires an explicit Hcore calculator (`initial_guess=None`)
rather than silently replacing another requested guess.

## Evidence and limits

- The independent 108-displacement private Cs46 gate passed at the exact native
  terminal frame: one unscaled `−571.052176 cm⁻¹` mode, 2/2 accepted, 4/4 energy/
  gradient and Hessian evaluations, 4385.38 s, maximum error `4.98e-7 Eh/Bohr²`.
- The new source-bound explicit policy on the same first geometry matches the
  independent occupied space (`0.9999999999999899`), energy and forces; 32 SCF
  iterations/36 Fock builds, `4.71e-13 Eh/Bohr` maximum force error, `93.57 s`.
  The source-bound native batch and saddle tests exercise refusal, same-frame
  native derivatives and ordinary warm/cold behavior. Receipts are under
  `source-bounded-cold-fock` in `/data/jzzeng/qualify-2844-2848-20261120`.
- The private Cs46 run does not qualify a public installed ECP saddle. ECP
  admission remains closed in both molecular adapters. The new bounded
  policy has no default/on-by-element behavior and does not assert general
  convergence or ground-state root selection.

Related: #2848 (uses #2846 and #2847).
