# Candidate: actual-grid iodide/methyl-iodide exchange control

Status: independent reference qualified; native consumer qualification pending
Date: 2026-11-13

## Purpose and boundary

This test-only I-CH3-I anion singlet is a distinct positive molecular saddle
candidate, not the original I-H-H or failed neutral I-H-I attempts, not a mixed
Ni/ECP fixture, and not the requested Ni reaction. Independent fixture discovery
does not supply a production preoptimization, electronic engine or numerical
Hessian implementation. No native optimizer or five-criterion search is claimed.

The model remains Gaussian/VWN-RPA B3LYP-D3(BJ), uncut two-body dispersion with
ATM off, canonical real-spherical def2-SVP and two I28 scalar ECPs, charge -1,
multiplicity 1, and actual native-default v4/level3 grid. The new carbon record
is a cohort-local version-1 export from the clean pinned BSE checkout, with
primary input/license/export hashes retained; the old canonical fixture is not
modified. Reference mass weighting uses I-127/C-12/H-1 isotope masses and no
frequency scaling or ECP-core mass subtraction.

## Actual-grid stationarity, not imposed chemical symmetry

`build/qualification/host-ich3i-stationary-discovery-20261113` retains the complete
attempt history. A bounded two-radial-parameter D3h discovery reaches a radial
root but exits 1 at the full Cartesian 2e-7 Eh/Bohr stationarity gate. The finite
angular quadrature does not exactly preserve 120-degree rotations: imposing
perfect chemical C3 symmetry leaves roughly 5e-6 Eh/Bohr nonradial forces.
Do not reinterpret a radial root as a full stationary geometry or change grids
to avoid this test.

A separate, bounded test-only correction retains the actual grid's reflection
symmetries and allows five coordinates: I z radius, the first H x position,
the paired H x/y positions, and the common I x displacement, with C fixed at the
origin to remove translation. Five fresh cold displaced gradients form a
Jacobian, followed by one cold corrected frame. The largest correction is
3.87e-5 Bohr, below the predeclared 0.005 Bohr cap; the full gradient maximum
becomes 1.18741e-8 Eh/Bohr. Six cold states and 78.03 seconds complete this
correction. SCF guesses never reuse saved/native/neighbor orbitals. The original
D3h failure and its source remain unchanged.

## Independent raw-H evidence and finite-difference order

The reference driver completes 108 cold displaced states, covering every raw
18-by-18 matrix column at 0.003/0.001/0.0003 Bohr. Its native source/SCF policy
does not change. Maximum independent orbital-gradient norm is 1.94e-10, minimum
occupied-root overlap exceeds 0.999950, and the largest overlap-metric electron
count error is 1.50e-13 for the required 60 electrons. All actual states and
unsymmetrized matrices are retained, not merely convergence flags.

Second-order raw antisymmetries are 2.658e-6/2.982e-7/4.171e-8 Eh/Bohr^2 and
translation residuals are 5.972e-7/6.842e-8/4.204e-8. The original driver exits 1:
even its finest raw matrix does not pass the unchanged 2e-8 symmetry gate.
Coarse-to-finest full-matrix differences are 4.715e-6 and 4.941e-7, consistent
with leading quadratic truncation, not a reason to loosen native raw gates.

A distinct read-only audit verifies all 108 saved states and reconstructs each
stored raw matrix exactly. It then uses the genuine fourth-order gradient
stencil, with h=0.001 Bohr:

```text
H[:,j] = (27*(g(+h e_j)-g(-h e_j))
           -(g(+3h e_j)-g(-3h e_j))) / (48*h)
```

This is `(9*D(h)-D(3h))/8`, not averaging H with its transpose or a native
matrix repair. No SCF or native electronic call runs in the audit. Its raw
antisymmetry 1.08379e-8 and translation 1.92923e-8 pass the original 2e-8 gates;
its difference from the separate 0.0003 Bohr raw matrix is 4.66402e-8, below
the unchanged 1e-5 independent raw-H comparison gate. The failed second-order
receipt is not relabeled as passed.

Only after raw inspection does ordinary mode analysis use its documented
symmetry handling and rigid projection. There are six rigid and twelve active
modes, exactly one negative frequency, -256.18589345 cm^-1. The remaining modes
are positive, starting at 94.88399440 cm^-1; no soft negative modes are deleted.
The imaginary mode is predominantly axial and changes the two C-I bond lengths
with opposite signs, as required for exchange rather than a minimum control.

## Native qualification remains distinct

The internal shared `NativeMolecularEvaluator` probe is bound to frozen source
`04226e07d04647fccf604519189b3e0a98815cf4b7ed6f37723030bec65390a6`
and its retained binary, without opening either public ECP saddle guard. It
performs its own native SCF/forces and must match the independent state before
its actual retained-frame raw Hessian and modes can be accepted. Its explicit
qualification-only primitive-work cap is one billion; no optimizer default or
scientific tolerance changes. A reference-only pass cannot promote this probe.

The CPU perf sampling attempt is denied by the host's `perf_event_paranoid=4`.
The setting is not changed; no profiler evidence is fabricated and no kernel
permission bypass or GPU work is attempted. The native endpoint's eventual
status, wall time and available work counters must remain visible separately.

The requested Ni TS, stationary mixed Ni/ECP mode fixture and practical
25--40-atom complete-electronic endpoint remain additional open obligations.

## Terminal native attempt

The actual six-atom/81-spherical-AO native attempt exits 124 at its finite
1800-second timeout. Complete measured wall time is 1800.02 seconds and maximum
resident size is 372000 KiB. It never returns its first complete energy/gradient
evaluation: the last published stage is `native-energy-gradient-start`. No native
orbital snapshot, raw Hessian, modes or optimizer result is accepted. The final
native/reference validator exits 1, explicitly rejecting the native endpoint.

The start receipt's 0.00496-second timestamp and zero work counters are only an
initial snapshot, not endpoint timing or evidence of zero SCF/Fock work. They
cannot establish whether SCF converged or which preparation/SCF/force phase
consumed the time. The Hessian primitive-work cap was not reached, so this is
not evidence that one billion Hessian primitives were exceeded. No source-level
hotspot diagnosis can be inferred from these incomplete counters or denied perf
sampling. The timeout is not memory exhaustion or proof that the saddle is absent.

The independent positive reference remains valid only in its own scope; this
cohort does not qualify the shared native I/ECP saddle consumer. Do not restart
this unchanged attempt with a longer timeout to manufacture the missing result.
The next useful work is source-bound phase/iteration/work instrumentation and
complete-endpoint CPU preparation/force improvements, not more unrelated tiny
fixture searches. Preserve this baseline and require fresh qualification under
any genuinely changed production source rather than restamping its receipts.
