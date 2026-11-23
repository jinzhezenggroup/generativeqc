# Decision: reuse scaled PBE response for the linked CPU PBE0 Hessian

Status: implemented (opt-in inventory)
Date: 2026-11-22

## Problem

The linked CPU Hessian originally admitted only canonical B3LYP. The
[generic PBE scalar-leaf experiment](../../rejected/2026-11-22-pbe0-host-hessian-maple-leaves.md)
linked but rejected a physical density tail during moving-grid response.
Compilable scalar leaves alone are not evidence of a complete endpoint.

## Decision

The extended `manifests/host_hessian_aot_catalog.json` binds the existing
scaled physical PBE0 point and directional evaluators to the shared CPU
Hessian. The RKS adapter consumes total density and Cartesian density
gradients. It applies the moving quadrature product rule for the source and
frozen mixed pullback; the induced response differentiates fields at fixed
points and weights. No new density floor or functional formula is introduced.
The fixed-grid PBE0 path streams directional coefficients rather than using
the B3LYP rho/sigma partial cache. Exact electronic MethodIR selects the same
integral-map family for admission and execution. The native total-method
binding adds named two-body D3(BJ) once without another electronic solve.

The default inventory remains B3LYP-only. The PBE0 extension is canonical
RKS, Direct FP64 and all-electron; it does not promote UKS, ECP, density
fitting, or modified electronic or correction graphs. Builds without this
binding keep the existing Python PBE0 path.

## Evidence and limits

The one-lane scaled-point checks cover the reported tail and the mixed
quadrature rule. Same-grid independent PySCF H₂ and HeH⁺ cases exercise the
raw electronic matrix and response. The corrected HeH⁺ and asymmetric water
tests compare dense and directional results with reconverged force differences
at three steps. The corrected-minus-bare action resolves the single dispersion
contribution. Water exercises a p-shell basis and a nine-iteration response.
Independent moving-grid PySCF references also check all 81 elements of bare
water's raw Hessian with spherical STO-3G and def2-SVP at two displacement
sizes. At `3e-4` Bohr the maximum errors were `1.12e-7` and `1.36e-7`
Hartree/Bohr², respectively. The def2-SVP case includes an oxygen d shell.
Distorted Cartesian STO-3G ammonia checks all 144 elements against the same
oracle, with maximum error `6.49e-8` Hartree/Bohr² at `3e-4` Bohr. All three
cases independently match the central energy, force, D/W, Hartree contraction
and combined semilocal XC/exact-exchange energy, and fail if the legacy Python
Hessian is invoked. These energy contractions do not certify the separate J/K
matrices or the induced Fock action in isolation.
The installed SDK, CLI and Python H₂ tests agree on energy, dense Hessian,
Cartesian action, completed directions and D3 work, with transactional
byte-capacity refusal in the SDK. The public and prepared entry points refuse
a nonconverged SCF state without publishing a derivative or retrying in Python.
Admission prices planned ERI work; completed visits can be lower after
skipping zero contractions and must not be reused as an exact work cap.
The ragged HeH⁺/water replay refuses an unsolved state, isolates an invalid
coordinate update, refuses the moved item's stale geometry, and compares
warm and recovered actions to fresh results. Complete builds now reject a
missing column dependency at configure time rather than advertising an
electronic identity whose corrected HVP would fail after SCF. The catalog
participates in source identity, and frontend parity checks its expected
linked total identities rather than skipping missing advertised methods.

An exploratory installed-build comparison used the corrected HeH⁺ fixture
above (spherical STO-3G, 8/4/8 grid, one BLAS/OpenMP thread). Over five
artifact-warm calls per arm, including fresh SCF and result publication, median
HVP/dense times were 1.55/2.81 s on the prior Python route and 0.037/0.119 s
on the extended route. Both used six SCF iterations and one/six response
solves, respectively; the largest output difference was below `4e-16`.
The native HVP/dense receipts counted 324/324 first and 486/486 curvature ERI
visits, 2048/12288 response XC point visits, and 3/18 D3 pair visits. The prior
route does not expose all matching counters. This is one warm comparison,
not a first-use or representative non-regression gate.

A second installed-build probe timed fresh public requests at both the
initial and moved geometries with warm artifacts (five samples per geometry).
For the H₂ B3LYP-D3(BJ) control, output and work receipts were identical and
all four median timing shifts were within 0.31%. For bare STO-3G water, HVP
medians were 2.28 s versus 0.17 s and dense medians 20.06 s versus 0.85 s;
both arms used 11 SCF iterations, 13 Fock builds and one/nine response solves.
The native HVP/dense paths counted 23004 first and 32886 curvature ERI visits
each, and 23040/205056 response XC point visits at the initial geometry.
The maximum cross-arm output difference over both geometries was below
`9e-15`. These fixture results do not qualify default inclusion.

For corrected spherical def2-SVP water, the completed paired HVPs agree
within `2.7e-15`; both routes use 12 SCF iterations, 14 Fock builds and one
12-iteration response solve. At both the starting and moved geometries the
artifact-warm legacy request takes about 32 s, versus about 1.44 s native.
The native HVP receipt reports 315722 first and 398786 curvature primitive
visits, 29952 induced XC point visits and nine D3 pair visits. The original
legacy campaign timed out before dense measurement; a separate bounded run
with the same warm artifacts measured one legacy dense call at each geometry.
Those calls took 339.56 s (initial) and 274.96 s (moved), compared with native
medians of 2.61 s at both geometries. Both arms used 12 SCF iterations, 14 Fock
builds, and 107 iterations over nine response solves. The raw matrices agreed
within `2.25e-14` Hartree/Bohr². The native dense receipt counted 315722 first
and 326255 curvature ERI primitive visits, 267264 induced XC point visits,
and 81 D3 pair visits; the legacy route lacks matching visit counters. The
single legacy sample at each geometry is not a tail-latency estimate.

Revisit default inclusion after installed first-use and existing B3LYP
non-regression coverage beyond the H₂ control. These checks are CPU-only and
do not close #2227 R3, R5, or R6.

The fresh-build default decision is recorded in
[the native CPU inventory promotion note](../compatibility/2026-11-22-default-cpu-hessian-inventory.md).
