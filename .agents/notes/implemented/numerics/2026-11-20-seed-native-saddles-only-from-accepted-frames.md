# Decision: Seed native saddles only from accepted frames

Status: implemented
Date: 2026-11-20

## Problem

An independently cold SCF can select a different root even at a small trial
displacement. The occupied-overlap gate correctly rejects that step; smaller
radii alone did not continue the matched NiO₂ branch. See
[the rejected cold retry](../../rejected/2026-11-20-repeat-cold-nio2-saddle-with-only-smaller-radius.md).

## Decision and invariants

Keep the cold policy by default. The all-electron molecular CLI and Python
adapter may opt into a new native SCF seeded by the last accepted frame's
bounded immutable density/geometry snapshot. Capture the seed after the raw
Hessian returns but before its derivative batch closes. Publish it only on
optimizer acceptance, never merely because a trial SCF or Hessian returned.
The initial frame starts cold. If a restored seed is unavailable, invalid, or
not actually consumed without fallback, fail closed and keep the last valid
geometry. Preserve the independent occupied overlap, MethodIR, grid, charge,
spin, residual and all other optimizer/final-verification gates. The snapshot
budget covers detached numeric storage, not aggregate provider work or RSS.

## Evidence and limits

The matched private NiO₂ B3LYP-D3(BJ)/def2-SVP endpoint passed 6 of 6 attempts,
fresh final derivatives and independent same-geometry occupied overlap
`0.9999999999999897`. The worst of three independent raw stencils differs by
`7.77e-7 Eh/Bohr²`; the single imaginary mode and unscaled frequencies agree.
Cold displaced reference SCFs also encountered a wrong root; reconverging
from the *independent* central occupied state, never the native coefficients,
preserved the comparison gate at all 54 displaced points. The bounded-native
snapshot implementation separately reproduced the descriptor-based NH₃ trial
and refused a joint root budget without publishing a successful trial.

Source-bound receipts: `ni-o2-warm-bounded-trial/independent-final-root-continuation`
and `ni-o2-native-snapshot-trial` under `/data/jzzeng/qualify-2844-2848-20261120/`.
The initial installed NiO₂ trial exhausted warm SCF at DIIS history 8, then
the underlying SCF recovered cold. The saddle refused that result. In an
installed energy-only diagnostic on the same trial geometry, history 16
converged warm without fallback in 30 iterations and matched the independent
occupied space to `0.9999999999999726`. DIIS depth is one fixed SCF control,
not a change to the Hamiltonian or a new retry policy. A one-geometry
diagnostic does not qualify a saddle trajectory; see the history-specific
receipts in `accepted-frame-saddle-production/ni-o2-installed-diis-scan`.
This does not grant ECP admission, universal convergence, or a lowest-energy
root guarantee. Revisit default selection only with broader independent
endpoint campaigns on cold, warm, rejected and failure paths.

Related: #2848.
