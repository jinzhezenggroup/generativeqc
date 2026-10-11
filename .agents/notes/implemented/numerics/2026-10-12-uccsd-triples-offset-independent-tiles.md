# Decision: lower unrestricted triples through offset-independent channel tiles

Status: implemented
Date: 2026-10-12

## Problem

The UCCSD solver supplies five independent amplitudes and physical spin-labelled
integrals, but restricted triples coefficients and pair coordinates cannot be
borrowed. A dense spin-orbital T3 would defeat the requested bounded-memory
contract. Embedding physical tile offsets in generated gather nodes would also
compile one artifact per orbital region and obstruct reuse.

## Decision

Use the real general spin-orbital connected/disconnected seeds, then expand the
independent three-term occupied/virtual cyclic sums. Enumerate conserved spin
assignments before constructing TensorIR. Evaluate only canonical aaa/aab/abb/bbb
patterns, with exact ordered-spin factors choose(3,nalpha)^2/36. Resolve every
mixed doubles permutation to the existing t2ab with its explicit sign.

Each graph input carries an immutable canonical-source/transpose/label binding.
The caller supplies local external coordinates while contracted axes stay whole.
Offsets are not scientific graph inputs or artifact identity. All equal-shaped
tiles use the same scalar graph and native CPU/CUDA lowering. The graph's JVP/VJP
can subsequently use these bindings for explicit cotangent scatter.

Real integral symmetry reuses the UCCSD ovvv/ooov/oovv feed inventory. There is no
new AO producer, PySCF production work or full spin-orbital/T3 representation.

## Rejected alternatives

- Four separately transcribed unrestricted slow formulas: correct independent
  oracles, but they duplicate spin/permutation policy across channels/backends.
- Restricted closed-shell r3 coefficients: wrong unrestricted physical dual.
- Full-spin/molecular T3 work arrays: useful only in a tiny test prototype, not
  a required production resource.
- Static physical gather positions in each tile graph: unnecessary per-region
  JIT identities. Explicit sliced-input bindings preserve mathematical ownership
  while permitting shape reuse and later reverse scatter.
- Claim the fixed-amplitude graph closes #1822: a converged complete owner with
  denominator, memory/work admission, failure policy and provenance is still
  required. Likewise local VJPs are not solved Lambda or nuclear forces.

## Evidence and invariants

The initial independent full-spin NumPy probe agrees with PySCF 2.14.0 unrestricted
slow triples for Li/OH/NH2/broken H2 to at most 1.8e-18 hartree. It is not retained
as a production fallback. Permanent tests compare channel reductions with the
independent slow inventory and its optimized total, not with that prototype.
H6 gives nonzero aaa/aab/abb/bbb coverage, partitioned tiles preserve channel
sums, and the restricted limit matches the existing RCCSD(T) definition.
All four native CPU tiles and all four JVP/VJP/finite-difference gates pass.
Mutated normalization and integral exchange signs fail the oracle criterion.

All four real-CUDA channel gates pass on n1 under finite Slurm job 7683
(37.61 seconds for the pytest run, not a complete energy benchmark). Raw host
and device qualification logs remain ignored under `.artifacts/`.

The final gate also covers asymmetric six-axis tiles on both native backends.
The source-matched host triples/provenance/UCCSD/compiler suite passes 191 tests
with seven explicit CUDA skips (39.70 seconds). Real-CUDA job 7687 passes all
four channel gates including asymmetric tiles (72.47 seconds for pytest).
The broken-symmetry H4 gate retains different alpha/beta coefficient matrices
and nonzero mixed channels. Ruff, CI-pinned ty, compiler structure and Sphinx
warnings-as-errors pass.

The new upstream identities have their own `uccsd_triples_source_manifest.json`.
The existing RCCSD manifest is part of frozen reference provenance and remains
unchanged; do not rewrite historical restricted evidence to add unrestricted
sources to it.

Retain independent unequal alpha/beta populations, five-block amplitude signs,
exact rational channel weights, real-integral symmetry and offset-independent
artifacts. Denominator/storage/work bounds and complete state response belong
to the future owner and must fail closed rather than silently spilling to CPU.

## Revisit when

A resident or shell-streaming owner changes feed ownership or tiled contraction
selection. Keep one scientific graph across CPU/CUDA and prove complete endpoint
semantics/work counts before promoting a different schedule.

## References

- #1822, #1825; UCCSD solver PR #2318 and equation PR #2316.
- `docs/developer/uccsd_triples.md`.
- `tools/generativeqc_cc/uccsd_triples_source_manifest.json` and `NOTICE`.
