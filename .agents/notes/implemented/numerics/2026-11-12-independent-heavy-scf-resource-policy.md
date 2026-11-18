# Decision: bind heavy-oracle resource and convergence controls explicitly

Status: implemented for qualification harnesses; Ni independent acceptance remains separate
Date: 2026-11-12

## Problem

The native canonical Ni/H UKS fixture explicitly allowed 600 SCF iterations,
but the independent-gradient helper still hardcoded 240. The original source
run returned a raw 12-by-12 native Hessian and two HVPs, then failed because
the independently cold-converged positive Ni-y displacement at 0.003 Bohr did
not pass its final convergence check under that separate policy. The source pytest
therefore remains failed, even though its native derivatives were retained.

## Decision

Independent solver helpers accept an explicit finite iteration allowance.
The heavy harness passes its declared native allowance to each independent
central/displaced solve and records the independent policy in the receipt.
The energy threshold does not change. A separate frozen reference-policy
manifest selects cold DIIS SCF and orbital-gradient tolerance `1e-11` for Ni,
with no extra nonlinear diagonalization after convergence. The actual physical
orbital gradient is explicitly recomputed and must remain below `1e-11` before
the reference is usable. The solver convergence flag is never overwritten.
Other fixtures retain the default extra diagonalization and `1e-10`. Physical acceptance gates do not
change. Independent solvers
continue to start cold at each geometry, with no native-density injection or
neighbor-state seeding. A convergence exception retains the actual independent
solver for failed density/orbital/occupation/residual evidence.

A new independent audit may compare retained native derivatives with new
references, but must identify the failed original run, hash its untouched
outputs, declare that no new native endpoint was executed and publish its own
acceptance result. The original receipt must never be relabeled. Installed
public endpoint acceptance still requires a separate audit of its own outputs.

## Rejected alternatives

- Accepting incomplete independent matrices because the native Hessian is
  symmetric or its own HVPs agree does not establish independent accuracy.
- Raising the orbital-gradient/density/raw-matrix gates hides a different
  failure and is not an iteration-budget correction.
- Restarting the expensive native Hessian cannot repair the independent
  solver's hardcoded budget.
- Disabling an extra check and accepting the **same failed state**, or marking
  its solver converged, would relabel a failure. A new cold attempt that retains
  its already converged density must instead independently pass the stricter
  actual physical-gradient check, plus all unchanged state/root/matrix gates.

## Evidence and boundaries

The failed source run is retained in
`build/qualification/host-canonical-heavy-v2-20261112/ni-uks/`, with actual
pytest exit 1. Native matrix execution took 7317.062 seconds, the two HVPs
610.343 and 610.043 seconds, with unchanged response gates. This is not a
passing independent, installed, TS or 25--40-atom qualification.

A distinct cold 600-iteration reference attempt also failed. Its retained
actual state stopped after **129** iterations with orbital-gradient norm
`3.0411971677982395e-10`, rather than exhausting 600. Inspection of the installed
PySCF 2.14.0 kernel identifies its additional unshifted diagonalization check;
raising the iteration cap alone cannot repair this early final-check failure.
A separate DIIS attempt tightened the orbital-gradient threshold to `1e-11`.
It passed the original failing y displacement but failed the positive Ni-z
0.003-Bohr displacement after 171 iterations, with final gradient norm
`3.107929544465008e-11`. Both raw failed states and source snapshots remain.
This rules out iteration exhaustion and shows that tighter DIIS alone, with
the extra nonlinear step retained, is not a complete correction. A proposed
plain Newton reference also failed a small H2 control and the independent Ni
central solve; that unqualified strategy was removed, and its failed runs
remain retained. The fresh no-extra-diagonalization reference must establish
its own convergence and strict actual physical gradient before root/full-matrix
qualification. It does not admit any failed state, change production SCF or
introduce an external molecular optimizer.

`tests/python/test_independent_scf_budget.py` checks the explicit allowance,
unchanged tolerances and preservation of a genuinely unconverged independent
state. These small oracle-policy tests are not heavy-element scientific gates.

## Independent stencil accuracy

The distinct no-extra-diagonalization audit v7 passed all 24 cold displaced
state/root checks for the first, 0.003-Bohr full matrix, with maximum actual
orbital-gradient norm `9.904445598528708e-12`. Its second-order reference
matrix nevertheless failed the unchanged `1e-5` raw-Hessian gate: maximum
error `1.7036358541444585e-5`. That entire audit remains failed.

A separately predeclared diagnostic selects the **worst** column, not a
favorable entry. Independent second-order column errors at 0.001 and 0.0003
Bohr are `1.8931008010092398e-6` and `1.7062154933356055e-7`: ratios
`0.1111212` and `0.0100151` relative to the coarse result, matching quadratic
reference truncation. Two additional cold points at +/-0.006 Bohr give a
fourth-order 0.003-Bohr column error `4.588340818401093e-10`. This diagnostic
passes its predeclared criteria but explicitly does **not** qualify full Ni.

The independent policy therefore selects fourth-order gradient stencils for
Ni at the same three displacement scales and the same raw-Hessian/HVP gates.
Other cases retain second order. Full qualification still requires all three
complete matrices, both native HVPs and every displaced root, not just the
diagnostic column. A new full audit can reuse the 24 individually validated
cold point states while separately counting 120 newly cold-converged points.
Neither reusing those raw points nor improving reference order reclassifies
the failed old matrix or changes the production derivative implementation.

## Complete retained-native and isolated installed Ni acceptance

On November 13, 2026, the distinct v8 audit actually exits zero after
3618.758 seconds. All 144 displaced states pass: 120 newly cold-converged
states and 24 independently root-rechecked cold v7 states. The three complete
fourth-order matrices at 0.003, 0.001 and 0.0003 Bohr have respective maximum
raw errors `6.2773164444e-10`, `2.8441515254e-9` and `6.9958525639e-9`, each
below the unchanged `1e-5` gate. Both retained native HVPs also pass every
independent matrix-action gate. The largest actual independent orbital-gradient
norm is `9.9408751277e-12`; the minimum occupied-root overlap is
`0.9954109865696475`. Its receipt explicitly records no new native execution
and keeps the original source pytest unaccepted.

The separately executed, isolated installed public Ni producer also actually
exits zero, after 9577.390 seconds with peak RSS 208328 KiB. Its fresh complete
native raw matrix and two HVPs take 7292.269, 609.604 and 611.007 seconds of
native execution, respectively. The supplemental cold central SCF converges
in 282 iterations/330 Fock builds at the unchanged tolerances. The producer
intentionally remains `accepted=false`, `independent_validation_pending=true`.

A new read-only installed-validation receipt separately accepts those outputs
against v8's matched central state and all three independent matrices. Density
error is `9.9534269715e-12`, energy error `2.9558577808e-12`, raw asymmetry
`4.2633119257e-11`, translation residual `4.6682546717e-11`, and raw matrix
replay error exactly zero. It verifies both HVPs and all native execution
contracts without new independent SCFs or modification of the producer.

The exact installed core SHA-256 is
`f3f2927c1c897af1a3304097b6caf5507cbe01302009d6b75500be0f58d89b55`, with
native source identity
`9f648efbe1d578fe5c7a5ecae11ce6ede0175bf1872cdb70b59e01382ac47656`.
The actual current production tree still matches that identity; no new
compilation or restamping of an older binary is involved. New evidence is in
`build/qualification/host-ni-independent-v8-20261112/` and
`build/qualification/host-ni-completion-recovery-20261113/`.

Original SCF240, v3--v7, Newton, stencil and source-pytest failures remain
failed. This acceptance closes this Ni raw-H/two-HVP reference and installed
endpoint gap only. It does not qualify an ECP molecular saddle, a stationary
mixed Ni/ECP fixture, practical complete 25--40-atom work or the requested Ni TS.
