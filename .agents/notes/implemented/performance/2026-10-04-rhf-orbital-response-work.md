# Decision: measure physical response and contract nuclear cross terms directly

Status: implemented, qualification in progress
Date: 2026-10-04

## Problem

The n2 PRO6000 230-AO ethane endpoint for #1829 spent 668.622 s in the combined
orbital/nuclear reference response. That aggregate does not establish that its
28 J/K actions dominate. Complete subphase measurements are needed before
promoting screened Z actions. The through-f provider may retain a bounded shell
derivative lease after canonical preparation; if that lease is not admitted,
three-pass polarization falls back to ordered public-AO derivatives. Record
which consumer actually runs rather than inferring it from angular momentum.

## Decision

Keep phase wall times and optional synchronized J/K census. Positive Schwarz
thresholds are provisional GMRES controls, defaulting to zero. The fixed-mask
API explicitly selects canonical complete ERI orbits, bypassing the ordinary
provider's density-dependent shell screening. Retain one immutable unscreened
source, independently audit the zero-screening physical Z residual at 1e-10,
and refine with exact GMRES if necessary. An additional live warm-start vector
is included in complete admission. Missing canonical capacity retains exact work.

For nuclear P:G'(D), the compiler emits the distinct-orbit sum of
`P_ab D_cd - P_ac D_bd/2`. Reuse existing canonical RSH derivative traversal,
full-range recurrence, public-to-Cartesian projection and translation
reconstruction. Only the first derivative output is consumed. Both operands
fit in the provider's pre-admitted two density slots; no four-index tensor or
additional scientific recurrence is introduced. Input, transformed-input,
weight, derivative/product and final-result finite audits precede publication.

Prefer this consumer over the generic and bounded through-f fallbacks. Retain
SPD's specialized three-pass path until a measured crossover supports changing
it, and keep an explicit selector for matched fallback checks. Shell/generic
pass counters identify the actual old consumer under the admitted budget.

## Work and memory

For N public AOs, **when the old generic consumer is selected**, it visits 3 N^4 ordered
quartets across the three polarization passes (before weight/center skips).
It already uses three-axis Dual3 jets and reconstructs the last distinct atom;
there is no additional 3A coordinate traversal factor in the current kernel.
Its jet count is the sum of u-1 over nonzero-weight ordered quartets in each
pass. This is traversal work, not a count of primitive FLOPs. The new source has
M=Nc(Nc+1)/2 Cartesian AO pairs and M(M+1)/2 canonical pair pairs. For each
nonzero-weight quartet with u distinct atoms, at most u-1 three-axis derivative
jets are evaluated; the last atom follows by translation. Canonical storage is
optional O(Nc^2) plus existing bounded pair metadata. The Cartesian expansion,
primitive multiplicity, angular recurrence cost and density cancellation prevent
converting these counts into a universal speedup or a FLOP rate. Bounded shell
fallback work is not the generic N^4 formula and remains unmeasured by this
new census; the pass counters establish which model applies.

The optional census adds two uint64 device counters, already admitted, and no
quartet-sized inventory. Profiling synchronizes calls and may select a different
SPD provider; complete comparisons must use matching controls and GPU UUIDs.
Census flags distinguish unavailable work from measured zero. J/K sub-times must
not be added to enclosing phase wall times.

## Rejected alternatives

- Setting the ordinary provider's threshold alone does not establish linearity:
  its generated shell path can depend on the current density.
- Passing the screened GMRES residual alone changes the scientific acceptance
  contract. Exact refinement and the independent physical audit are mandatory.
- Treating the aggregate 668.622 s as J/K time is unsupported before measurement.
- Three energy differences are a valid bounded fallback, but repeat expensive
  derivative source work and introduce subtractive cancellation.
- New recurrence code, an N^4 derivative tensor or a CPU oracle in production
  would violate ownership or capacity requirements.

## Evidence

Validation and matched endpoint measurements are being retained under
`.artifacts/orbital/` and the associated PR. Independent tests include signed
fixed-mask linearity/self-adjointness, dense masked libcint J/K, exact correction
for aggressive masks, all repeated-index bilinear coefficient orbits, spherical
and Cartesian through-f nuclear derivatives, two finite-difference steps,
nonfinite publication refusal, HF-limit fallback parity and complete DF forces.

This work does not qualify the outstanding large source-factor
`atol=rtol=3e-10` gate, certify global RHF stability, or resolve #1829's separate
cold-force pair discrepancy. Small-system closure is not large-system force
qualification. Actual measured results must be attached before performance
promotion.

## References

#1808, #1809, #1818 and #1829; `docs/developer/df_ccsdt_gradient.md`.
