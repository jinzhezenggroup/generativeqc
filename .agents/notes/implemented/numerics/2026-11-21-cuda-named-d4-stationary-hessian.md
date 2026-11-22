# Decision: Admit standard D4 Hessians independently of electronic spin and exchange

Status: implemented
Date: 2026-11-21

## Decision

Extend [the qualified PBE-D4 composition](2026-11-21-cuda-pbe-d4-stationary-hessian.md)
instead of adding hybrid- or UKS-specific dispersion differentiation. The
electronic provider must already match the authenticated native stationary
model. Borrow the primal standard-profile D4 plan and use the physical total
molecular charge (not the spin populations) for its EEQ action. The action
does not allocate gCP scratch, add another CPKS solve, or require a new Python
receipt ABI. The final device gate still precedes publication. The existing
atom limit and aggregate device arena bound are retained.

Python selects exact catalog identities or the intrinsic restricted PBE-D4
owner. Catalog membership alone is not electronic admission. r²SCAN-D4 uses
and checks the same variational SCF domain as bare r²SCAN; r²SCAN-3c requires
the separate D4 profile and gCP. The CLI rejects unqualified catalog D4 models
before SCF. Calling an independent Python D4 implementation in production or
rebuilding a fit from the name would break the primal/correction identity.

## Evidence

On RTX 5090 (Slurm job 1668), 16 complete named Python/CLI cases cover PBE,
PBE0, B3LYP and r²SCAN with Direct and full-rank DF, for bent neutral STO-3G
water RKS and its charge +1 doublet UKS. The same job passed both existing
PBE-D4 frontend regressions. Subtract each bare CLI matrix, then compare with
two Richardson-extrapolated displacements of independent `dftd4==3.7.0`
gradients: the worst correction error is `6.92e-16` Hartree/Bohr². The tests
check the independent D4 energy, a nontrivial charge effect for UKS, raw
symmetry and translation nulls, and Python HVP versus CLI matrix action. The
dense receipt includes all nine correction directions; the HVP completes one
direction. This component comparison does not independently requalify each
electronic model. The source-matched build is the local worktree based on
`f8e0945d0`. Job 1669 also passed the two independently reconverged PBE-D4
electronic-plus-correction benchmarks and the targeted D3/r²SCAN regressions.
The final matched build (job 1671) repeated the 16 component cases, including
PBE1PBE and B3LYPG aliases matched by catalog identity rather than name, the
two legacy PBE-D4 cases, and rejection of an unsupported electronic graph or
modified fit. Local evidence is `build/hessian-gpu-sm120/d4-final-matched.xml`,
`d4-final-matched-gpu.log`, `d4-general-regression-gpu.log` and the
`d4-general-cache-*.log` / `d4-alias-cache-*.log` receipts.
