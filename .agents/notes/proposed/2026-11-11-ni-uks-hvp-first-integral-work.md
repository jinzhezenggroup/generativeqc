# Ni UKS HVP: qualify the frozen first-integral schedule before response tuning

Status: proposed
Date: 2026-11-11

## Matched state, not a Hessian receipt

The asymmetric neutral NiH3 doublet uses Ni `(0.01,-0.02,0.03)` and H
`(1.8,0.1,0.03)`, `(-0.5,2.4,0.3)`, `(-1.7,-0.8,1.1)` Bohr. It retains
canonical real-spherical def2-SVP, B3LYP/VWN-RPA, Direct/no DF and the native
default Treutler grid. The source basis remains
`tests/data/global_hybrid/manifest.json`; there are 46 AOs, occupations `(16,15)`
and 50,064 grid points. This does not supersede the failed NiH/NiH3 roots in
the rejected ECP-grid note.

At native source `fe9ebbdff122d8d63974be0bd180fd0b3dd1c2f5`, cold SCF takes
123 iterations, giving `-1509.5720618893429 Eh` and physical residual
`1.25719545045e-13`. An independently cold atomic-guess reference matches
density within `7.8727580011e-12`, energy within `2.7284841053e-12 Eh`, and
has orbital-gradient norm `4.0034411435e-11`. Coupled induced-Fock actions
from each spin block match the independent UKS response within
`3.3135993949e-13` and `2.9872632146e-13`; both cross-spin actions are nonzero.
These checks certify that state and those actions, not a nuclear HVP.

The default 128 MiB grid workspace rejects this fixture before response. An
explicit bounded 2 GiB diagnostic workspace binds successfully. Do not promote
the public capability or change the default budget from this fact alone.

## Bounded negative endpoint and observed hot path

A complete low-level HVP under GDB reaches its 1,800-second wall-time limit
without returning a value. Peak process RSS is 2,034,956 KiB. This is neither a
successful HVP nor a measured failed GMRES solve. No response iteration count
or residual can be inferred from the last `native-hvp-start` marker.

A separate interrupted profile, with a newly converged native state matched
to the independent central density and exact saved grid, stops in
`bounded_range_moments<14>` under `factorized_eri_records`. Its Python stack is
`CompositeRKSResponse` perturbation -> directional first-integral contraction
-> `_FirstDerivativeProvider.raw_blocks` -> `FactorizedERIHost.raw_first`.
An entry breakpoint confirms operation `1`, not weighted second-order
operation `2`. Thus response conditioning is not the first demonstrated
execution bottleneck.

## Candidate schedule changes, not accepted performance claims

The current record consumer calls the complete factorized primitive for every
Cartesian component. Each call rebinds identical primitive-pair geometry and
Boys/range moments; those depend on the record and quartet angular bound, not
the Cartesian component. Prebinding them once per record is a bounded exact
schedule candidate, not permission to screen primitives or weaken accuracy.
The raw-first path also computes directional second-order fields it does not
publish. Coupled alpha/beta nuclear RHS construction repeats raw first
integrals while changing only density weights. These are separate reuse
opportunities; establish endpoint evidence rather than assuming a kernel win.

Preserve independent mathematical slots, normalized Cartesian/spherical
weights, late-failure atomic publication, finite compiler/cache ownership and
explicit numeric leases. Compiler ownership must remain in the compiler; a
runtime template may only bind/stream generated programs. A reused geometry
must have one invocation's record lifetime, never an unchecked persistent
identity cache.

## Still required

Freeze and retain state, response, raw symmetry, translation, H@v and independent
multistep-gradient gates before accepting any Ni result. One displaced cold
reference fails under the graph-ordered XC protocol, despite matching native
SCF convergence; failed 200/600-cycle and independent Newton attempts are
retained locally. A separately cold, explicitly written canonical XC reference
then passes on the exact native displaced grid: density error
`2.0496937481e-11`, energy error `1.4324541553e-11 Eh`, orbital-gradient norm
`1.9410107695e-10`. No native/reference density seeding or gate relaxation is
used. All six displaced points at the three frozen step sizes now match their
native states; this still does not supply a native HVP. Do not silently drop
the failed protocols or reinterpret an independent oracle as production work.

Raw HVP/full Hessian, total D3 curvature, mixed Ni/I/Cs ECP forces and second
order, and a fixed-method molecular TS task remain separate open qualifications.

## Geometry-only schedule follow-up

The record-local geometry change is independently validated and documented in
`../implemented/performance/2026-11-11-factorized-host-record-geometry-reuse.md`.
Its complete Ni trial nevertheless reaches a 3,600-second limit without an HVP
and peaks at 2,034,812 KiB RSS. This negative result does not establish a GMRES
failure or a unique remaining hotspot. Scalar-direction lowering and shared-spin
RHS experiments are separately tracked in
`2026-11-11-factorized-directional-first-and-shared-spin-rhs.md`; retain their
distinct source identities and gates rather than restamping the geometry-only
run as their endpoint measurement.

## Local evidence

### Post-response lazy compilation remains visible

The frozen compiler-only point-panel experiment subsequently terminates at its
3,600-second limit with exit status 124 and peak RSS 2,258,664 KiB. Its explicit
response marker records 46 actions in 442.130459 seconds, but no native HVP,
returned response residual or full Hessian follows. The separate timeout receipt
is `/tmp/gqc-ni-hvp-fxc-frozen-probe/timeout-receipt.json`; source provenance
remains the frozen `bfd1bb6cc8` checkout plus its recorded experiment scripts.

Read-only process inspection on 2026-11-11 also observes both the spin-gap
and frozen point-kernel trials compiling component/coordinate-specific
second-order sources after their explicit response-complete markers. The
compiler command uses the verified per-invocation sccache owner and the
existing `second-cache-cpu/native/.native-runtime-*` slots; this is not a
hidden reference-engine operation. One live generated primitive validates two
exponents, nine center coordinates and eight weights, and its emitted C++ is
approximately 488 kB. Do not assume the generic four-center ERI owner removes
the expanded lowering of every other integral family.

The read-only cache snapshot `/tmp/gqc-ni-second-jit-inventory.json` finds
271 completed second-order artifact records and matches every recorded source
SHA256 to retained generated source bytes. It separates original-worktree and
frozen-snapshot include roots. This is a shared directory containing multiple
protocols and earlier attempts: summed compiler durations are overlapping
historical work, not one endpoint's wall time, and the count is not an exact
per-HVP attribution. The snapshot therefore demonstrates ongoing expanded
compilation, not a unique causal explanation for the entire timeout.

Before another long endpoint trial, instrument source/contribution boundaries
and audit component/coordinate specialization in non-ERI second-order
consumers as well as the first RHS. Reuse canonical compiler algebra and keep
the existing bounded fallback; do not replace derivatives with finite
differences, disable source identities, alter include roots merely for cache
hits, or attribute response-only timing to a complete HVP. A `py-spy` attach
was denied by host permissions; do not repeat it expecting a Python stack
receipt from this environment.

`/tmp/gqc-continue-ni-hvp-gdb.log`, its adjacent `*-time.txt`,
`/tmp/gqc-continue-ni-hvp-profile-gdb.log`,
`/tmp/gqc-continue-ni-independent-hvp/` and
`/tmp/gqc-continue-ni-reference-newton/` preserve the unabridged diagnostics.
The frozen gate-file SHA256 is
`8f15ecb93a8095c07861ad85e24498e670323306a8f5578ae3201d88a96cda95`.

## Subsequent complete experimental action

The committed one-electron source reuse plus still-scratch directional/first-only/
shared-spin lowering now returns one complete NiH3 HVP within its finite limit
and passes every pre-frozen independent action comparison. See
`2026-11-11-ni-complete-experimental-hvp-acceptance.md` for the sealed provenance,
all three errors, work and process timings. Production integration, native full
raw matrices and installed qualification remain open; earlier failures are
retained, not relabeled as successful runs.
