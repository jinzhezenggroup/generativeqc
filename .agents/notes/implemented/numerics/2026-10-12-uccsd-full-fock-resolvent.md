# Decision: bounded full unrestricted triples Fock resolvent

Status: implemented
Date: 2026-10-12

## Problem

The corrected Lambda milestone in #2347 supplies complete amplitude response,
but diagonal eps-to-Fock bars alone cannot make the canonical (T) Lagrangian
stationary under same-space orbital rotations. The resolvent derivative
`d D^-1 = -D^-1 (dD) D^-1` acts on every occupied/virtual slot, including
off-diagonal same-spin oo/vv elements. Dividing internal orbital gaps would
hide this missing chain and fail at degeneracy. The existing restricted
compiler already carries an explicit separable-resolvent contract.

## Decision

Extract bounded X=C/D and Y=(C+V)/D from the actual shared unrestricted
nine-term builder. Preserve the scalar energy graph and independent mixed-spin
populations; no second inventory, closed-shell coefficient or full molecular
spin-orbital T3 enters production. Contract paired tiles over five common
coordinates, leaving one spin-owned axis open on each side. Symmetric slot
moments have occupied-minus and virtual-plus signs and retain the actual
1/36 or 1/4 channel factor exactly once.

The new consumer borrows a live triples-response owner and freshly replays
the complete parent/source response. It sweeps all six axes and uses triangular
open-axis page pairs, scattering off-diagonal blocks and their transposes once.
Only two X/Y pages are retained. Right pages are recomputed rather than storing
the complete axis grid. For n open-axis pages and m common-coordinate tiles,
both vector-page evaluations and moment calls number m*n*(n+1)/2. The planner
accounts for actual regular/edge page shapes and their recomputation weights,
not a resident-memory proxy for work.

Admit fresh source response, all paired/recomputed native scalar-loop work,
retained/publication buffers, two X/Y pages, binding copies and native staging
before execution. Numeric/executor fallback deterministically reduces pages;
complete pair and total-work bounds remain hard gates. These are logical numeric
reservations, not allocator-enforced RSS or cache/context/IR metadata bounds.
Host compensation work is not represented as native scalar-loop FLOPs, and no
complete-method performance claim follows from this bounded algorithm.

The full moments replace the diagonal-only denominator contribution. Every
completed response must match their diagonal to freshly differentiated eps
bars. Keep all spin-owned source bars and return separately immutable moments.
Late failures strip even the already replayed correction's energy/bars from
the combined failed record; no prefix endpoint is published. Borrowed lifetimes
and admitted configuration fail closed, and closing this owner closes neither
the triples consumer nor the parent/backend.

## Rejected Alternatives

- Diagonal denominator bars plus an ov-only physical gate: hides oo/vv response.
- Internal-gap division: undefined at degeneracies and not the resolvent chain.
- Full molecular T3 or complete-axis X/Y retention: violates bounded storage.
- Page shrinking without complete recomputation admission: memory-bounded is
  not work-bounded.
- Add full moments on top of existing EPS diagonals: double-counts denominator
  derivatives.
- Duplicate connected/disconnected equations: permits scientific drift from
  the already independently qualified energy.

## Evidence and Gates

`tests/python/test_uccsd_triples_fock_response.py` expands the pinned independent
triangular PySCF GCCSD(T) oracle only in tests. It compares actual native X/Y
and all four channels/six axes at 3e-13, including unequal populations, unequal
paired-page lengths and exact internal degeneracies. Independent dense-test
Fock diagonalization gives off-diagonal resolvent finite-difference closure at
2e-9, rather than merely reusing the production marginal formula.

Complete broken H4, asymmetric H6 and NH2 consumers compare with independent
full-spin molecular moments at 2e-10 and preserve the already qualified EPS
diagonals at 3e-13. Tests cover aggregate admission, deterministic fallback,
late vector/moment/nonfinite failures, forged fresh EPS bars, immutable results,
borrowed closure and empty channels. Real CUDA qualification requires finite
Slurm and forbids native CPU equation execution. A one-off twelve-graph
pre/post energy-identity comparison confirms the extraction preserves existing
scalar mathematics; no production-function fingerprint is added to required CI.

## Remaining Boundary

Full Fock cotangents are not complete corrected physical or nuclear response.
The standard physical owner continues rejecting corrected Lambda records.
Next compose explicit corrected Hamiltonian/UHF/nuclear owners, retain actual
all-rotation stationarity, and qualify reconverged complete-force finite
differences, valid independent analytic references and restricted RCCSD(T)
force parity. Neither #1825 nor the full CPU/CUDA goal is complete here.

## References

- #1825, #2340, #2347.
- `docs/developer/uccsd_triples.md`.
- `2026-10-12-uccsdt-corrected-lambda.md`.
- `python/generativeqc_compiler/cc/triples_fock_response.py`.
