# Decision: Pack symmetric AO pairs for the fitted spectral tangent

Status: implemented
Date: 2026-11-21

## Problem

The [dense spectral schedule](2026-11-21-cuda-df-hybrid-spectral-response.md)
admitted RKS but rejected the same 48 AO, 226 auxiliary size in UKS. The old
PBE0 water dimer cation reached the one-billion tiled-product cap in RTX 5090
Slurm `1779`; its layout would also exceed the 64 MiB fitted-owner budget.
Solving both orientations of every symmetric raw AO pair did duplicate
auxiliary work. The UKS cotangent also separately contracted two moving-density
terms related by reflection.

## Decision

Apply `δA V + A δV` to triangular rows only, averaging the two raw orientations
before applying the inverse. Both spin components share the expanded result.
The generated map keeps the same auxiliary reduction order and does not assume
an occupied-density or positive-density representation. For the three-center
exchange weight, contract `D δB D + 2 δD B D` in one orientation and project
onto the symmetric AO domain. This retains all three product-rule terms.

The packed stage borrows map workspace, and the expanded tangent borrows the
staged three-center direction slot until its last read and subsequent output
copy in the same stream. The dense seed and inverse tangent share a slot in
nonoverlapping phases; the ordered spectral term borrows rotation scratch.
The affine emitter permits an opt-in lane-local add to overwrite an internal
operand at its last use. Retained outputs, input buffers, permutations and
contractions cannot serve as in-place writes. Status gates remain in place
before consuming staged results or publishing caller outputs. Sizes, packing
visits and padded tile work are checked before submitting CUDA work.

## Qualification

At 48 AO and 226 auxiliary functions, the packed UKS hybrid plan admits
66,871,296 bytes including the raw panel (64 MiB cap). Its reference plan
charges 13,108,925 scalar visits and 667,201,536 padded product slots. Its
direction charges 51,114,609 scalar visits and 912,310,272 padded slots,
down from 1,091,321,856 padded slots. These are fitted-algebra admission
counts rather than complete endpoint work or performance metrics.

RTX 5090 Slurm `1780` completed the open-shell PBE0 water dimer cation HVP in
51.35 s: one direction, 36 induced Fock actions, 17 response iterations, one
DF value pass and one DF contraction pass. The spherical cc-pVDZ/full
def2-universal-JKFIT calculation discards 12 roots at relative cutoff
`3.636967353327937e-6`. An independent PySCF/Libxc UKS oracle rebuilt expanded
fitted ERIs and reconverged at every displaced geometry. Successive Richardson
estimates differ by `2.92e-8 Ha/Bohr²`; the two measured projected errors are
`3.10e-8` and `2.20e-8 Ha/Bohr²` against an absolute `2e-7` gate. Translation
passes; there are no CPU numerical fallbacks, intermediate panel downloads or
displaced SCF evaluations in the native action.

The tightly converged independent SCF uses Newton refinement when DIIS alone
does not converge, and checks the orbital gradient at every geometry. Logs,
source-matched checksums and the independent replay are retained under
`build/hessian-gpu-sm120/r2scan-orbital-f/` (`uks-packed-binaries.sha256`,
`uks-packed-work-estimate.log`, `probe-pbe0-uks-dimer.py`, and
`pbe0-uks-dimer-{baseline,endpoint}-n4.log`). Slurm `1781` passed 74 native
and independent-oracle tests and 17 public endpoint regressions. The packed
RKS endpoint in `1783` retains the independent reference agreement.

## Full-rank follow-up

Slurm `1783` also confirmed the full-rank UKS owner still exceeded its cap
with depth-ordered evaluation. Unlike the spectral Gram, the full-rank
Gram depends on fitted tangents. Prioritizing the frozen Fock instead
consumes the implicit-fit residual before materializing the dressed fitted
tangent; this is a schedule change only. It reduces the full-rank UKS
admission at 48 AO/226 auxiliary functions from 69,348,608 to 61,017,344
bytes including retained raw A, with 20,513,179 scalar visits and 780,779,520
padded slots per action.

The same cation retaining all 226 roots passed on SM120 Slurm `1784` in
51.39 s, with 36 induced Fock actions, 17 response iterations, one DF value
pass and one DF contraction pass. Two projected errors against the
independently reconverged UKS energy oracle were `4.36e-8` and
`2.69e-8 Ha/Bohr²`. The translation and residency gates also passed. See
`pbe0-uks-dimer-fullrank-final-n4.log` and `uks-fullrank-schedule-work.log`.
No architecture other than SM120 is qualified by these measurements.
