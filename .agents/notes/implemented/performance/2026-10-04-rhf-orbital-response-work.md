# Decision: preserve shell reuse in physical nuclear response

Status: implemented; two-pass schedule qualified at the recorded endpoint
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

The default nuclear schedule is symmetric polarization
`[E2'(D+P)-E2'(D-P)]/2`, using the same admitted unscreened provider in both
passes. E2 is homogeneous quadratic in its density, so this removes one
traversal without changing the source or losing shell-level primitive/component
reuse. Reuse the old combined host matrix after the first consumer drains;
the existing three-pass capacity bound still covers all live payloads. Keep
legacy three-pass polarization as a validation selector.

For the separate experimental canonical P:G'(D), the compiler emits the distinct-orbit sum of
`P_ab D_cd - P_ac D_bd/2`. Reuse existing canonical RSH derivative traversal,
full-range recurrence, public-to-Cartesian projection and translation
reconstruction. Only the first derivative output is consumed. Both operands
fit in the provider's pre-admitted two density slots; no four-index tensor or
additional scientific recurrence is introduced. Input, transformed-input,
weight, derivative/product and final-result finite audits precede publication.

Do not enable this canonical consumer by default: the measured bounded-shell
crossover below rejects that promotion. Keep it as an explicit experiment over
generic/bounded through-f consumers, preserving specialized SPD leases.
Shell/generic pass counters identify the actual old consumer under the admitted
budget. The ordinary two-pass schedule preserves those leases in every domain.

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
- Enabling bounded through-f **value** dispatch just because the nuclear
  derivative lease is useful would repeat a rejected promotion. The existing
  [value-policy decision](2026-10-02-through-f-value-policy.md) retains separate
  opt-in selection after measured complete SCF regressions in #1666. Derivative
  reuse does not establish value profitability. Those RTX5090 measurements are
  evidence against blind promotion, not a timing calibration for this PRO6000.

## Measured rejection of canonical AO promotion

On n2 PRO6000 UUID `GPU-54595246-dbdc-a633-dc38-7bd8eea3831a`, Slurm job2226,
source211d9fce2, the same frozen binary/input measured:

- Three-pass shell polarization: complete force1352.438s; response668.733s,
  including J/K512.631s and nuclear two-electron155.771s.
- Canonical bilinear: complete force1486.362s; response771.432s,
  including J/K512.460s and nuclear two-electron258.643s.
- The independent RHF start differed by31.178s; the102.872s nuclear regression
  remains after separating that variability. One pass was slower than three.
- The old consumer was three **shell** passes, not the generic N^4 fallback.
  Its internal quartet/jet work was not measured and must remain missing.
- Canonical counts were575639415 quartet visits and1276669675 three-axis jets.
  These match the geometry-derived bounds and are not FLOPs.
- Both force results pass the retained independent directional FD gate3e-7 at
  h1e-4 and3e-5. Their cold-pair force difference2.5505e-9 passes the unchanged
  extra3e-9 gate. The canonical path is numerically qualified here but rejected
  for performance. Energy-only is325.026s; subtracting cold totals is not an
  exact incremental force cost.

Amdahl bounds from the measured baseline are1.130x for eliminating the whole
nuclear two-electron phase, and1.481x for eliminating the entire Z solve.
These are idealized limits, not achieved speedups. Fixed screening remains an
experiment with a mandatory unscreened audit; the default threshold is zero.

## Measured two-pass shell result

Source8a8375983, n2 job2238 uses the same GPU UUID as job2226, but a separately
frozen binary/allocation. Its complete force is1306.786s, orbital response616.828s
and nuclear two-electron103.814s. The nuclear phase improves1.500x, saving51.957s.
Raw complete improvement is1.035x (45.652s); cold RHF increases6.158s and is not
assigned to the nuclear schedule. J/K is512.691s, essentially unchanged. The
actual consumer is now two shell passes; complete numeric capacity remains
7170696363bytes. Internal shell quartet/jet work remains unmeasured.

The two-pass Z residual is1.360e-13, stationarity1.139e-11, and independent
directional FD errors4.896e-9/5.620e-9. Cold force differences from legacy and
canonical are1.119e-9/1.850e-9, both below the unchanged3e-9 gate. These gates
qualify this schedule comparison, not the outstanding source-factor or global
stability requirements.

The same-binary 1e-12 screening control completes in 1350.010 s, with response
616.589 s, Z solve 439.312 s and nuclear response 103.812 s. It removes only
4924248 of 16117903620 complete J/K ERI evaluations (0.03055%); canonical visits
are unchanged. There are 24 provisional actions, four exact actions, 12
iterations and no corrective solve. The unscreened residual is 1.358e-13.
The 0.036 s Z-time difference is not a demonstrated speedup. Cold RHF increases
43.335 s, explaining almost all of the increased complete time. Keep screening
at zero; do not promote this threshold based on subsecond single-sample noise.

Across all four retained force schedules, all six cold-pair comparisons pass
3e-9 (maximum 2.702e-9), and all independent two-step directional FD gates pass.
Job2238 finishes with exit0. No additional GPU allocation is left running.

## Evidence

Validation and endpoint measurements are retained under
`benchmarks/results/rhf-orbital-response-20261004/` with frozen artifacts under
`.artifacts/orbital/`. Independent tests include signed
fixed-mask linearity/self-adjointness, dense masked libcint J/K, exact correction
for aggressive masks, all repeated-index bilinear coefficient orbits, spherical
and Cartesian through-f nuclear derivatives, two finite-difference steps,
nonfinite publication refusal, HF-limit fallback parity and complete DF forces.

This work does not qualify the outstanding large source-factor
`atol=rtol=3e-10` gate, certify global RHF stability, or resolve #1829's separate
cold-force pair discrepancy. Small-system closure is not large-system force
qualification; the large endpoint and its specific acceptance limits are
recorded separately above.

## References

#1808, #1809, #1818 and #1829; `docs/developer/df_ccsdt_gradient.md`.
